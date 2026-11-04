"""Labeling backends: a geometric baseline and two VLM APIs behind one interface.

  heuristic  3D boxes + HD map only, no image. The "SQL over perception output"
             baseline a VLM has to beat on the semantic labels.
  gemini     Google Gemini via google-genai, with schema-constrained JSON output.
             Set GEMINI_API_KEY. (Waymo's EMMA research model is built on Gemini.)
  openai     Any OpenAI-compatible endpoint: OpenAI itself, or a local open model
             such as Qwen2.5-VL served by Ollama (--base-url http://localhost:11434/v1).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass

from .features import object_features
from .prompts import ImagePart, TextPart, VLMRequest


@dataclass
class VLMResult:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Backend:
    name = "base"
    cacheable = True  # API backends: never pay twice for an identical request

    def __init__(self, model: str):
        self.model = model

    def supported_labels(self, labels: list[str]) -> list[str]:
        return labels

    def generate(self, req: VLMRequest, temperature: float) -> VLMResult:
        raise NotImplementedError


class HeuristicBackend(Backend):
    """Geometric proxies for the human-labeled classes.

    Waymo's 3D label set has no 'cone' class, so this keys on sign/static objects
    sitting on or next to mapped lanes. It is deliberately simple: the point is a
    floor to compare against, and a cheap first-stage filter in a cascade.
    """

    name = "heuristic"
    cacheable = False
    LABELS = ("construction_zone", "road_change", "emergency_vehicle")

    def __init__(self, model: str = "geometry-v1"):
        super().__init__(model)

    def supported_labels(self, labels: list[str]) -> list[str]:
        return [l for l in labels if l in self.LABELS]

    def generate(self, req: VLMRequest, temperature: float) -> VLMResult:
        t0 = time.perf_counter()
        static = [f for f in object_features(req.sample)
                  if f.in_camera and f.type == "sign" and f.range_m <= 50 and f.lane_dist_m is not None]
        on_lane = [f for f in static if f.lane_dist_m <= 1.5]
        near_lane = [f for f in static if f.lane_dist_m <= 3.0]
        out = {
            "construction_zone": {"present": len(near_lane) >= 2, "confidence": min(1.0, 0.2 * len(near_lane))},
            "road_change": {"present": len(on_lane) >= 1, "confidence": min(1.0, 0.25 * len(on_lane))},
            "emergency_vehicle": {"present": False, "confidence": 0.0},
        }
        labels = {k: v for k, v in out.items() if k in req.labels}
        return VLMResult(json.dumps({"labels": labels}), time.perf_counter() - t0)


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", thinking_budget: int | None = None,
                 base_url: str | None = None):
        super().__init__(model)
        from google import genai  # pip install -e '.[gemini]'
        from google.genai import types

        # Reads GEMINI_API_KEY (or GOOGLE_API_KEY). base_url is for proxies and tests.
        self.client = genai.Client(http_options=types.HttpOptions(base_url=base_url) if base_url else None)
        self.thinking_budget = thinking_budget

    def generate(self, req: VLMRequest, temperature: float) -> VLMResult:
        from google.genai import types

        contents = [p.text if isinstance(p, TextPart) else types.Part.from_bytes(data=p.data, mime_type=p.mime)
                    for p in req.parts]
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=req.schema,
            thinking_config=(types.ThinkingConfig(thinking_budget=self.thinking_budget)
                             if self.thinking_budget is not None else None),
        )
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        usage = resp.usage_metadata
        out_tokens = None
        if usage is not None:
            out_tokens = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        return VLMResult(resp.text or "", time.perf_counter() - t0,
                         usage.prompt_token_count if usage else None, out_tokens)


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None):
        super().__init__(model)
        from openai import OpenAI  # pip install -e '.[openai]'

        # Local servers (Ollama, vLLM) ignore the key but the client requires one.
        key = os.environ.get("OPENAI_API_KEY") or ("local" if base_url else None)
        self.client = OpenAI(base_url=base_url, api_key=key)

    def generate(self, req: VLMRequest, temperature: float) -> VLMResult:
        content = []
        for p in req.parts:
            if isinstance(p, TextPart):
                content.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                url = f"data:{p.mime};base64,{base64.b64encode(p.data).decode()}"
                content.append({"type": "image_url", "image_url": {"url": url}})
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        usage = resp.usage
        return VLMResult(resp.choices[0].message.content or "", time.perf_counter() - t0,
                         usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None)


def make_backend(name: str, model: str | None = None, base_url: str | None = None,
                 thinking_budget: int | None = None) -> Backend:
    if name == "heuristic":
        return HeuristicBackend()
    if name == "gemini":
        return GeminiBackend(model or "gemini-2.5-flash", thinking_budget, base_url)
    if name == "openai":
        if not model:
            raise ValueError("--model is required for the openai backend (e.g. gpt-4o-mini, qwen2.5vl:7b)")
        return OpenAICompatBackend(model, base_url)
    raise ValueError(f"unknown backend {name!r}")


def generate_with_retry(backend: Backend, req: VLMRequest, temperature: float,
                        attempts: int = 4) -> VLMResult:
    """Retry transient API failures (rate limits, 5xx) with exponential backoff."""
    for attempt in range(attempts):
        try:
            return backend.generate(req, temperature)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")

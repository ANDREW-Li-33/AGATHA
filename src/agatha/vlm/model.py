"""The model interface, Claude behind it, which can also do a dry run that sends nothing, a local open-weight model
behind vLLM, and a fake for tests.

Content blocks are model-neutral: text as Claude takes it, and images as {"type": "image", "ref": <path
under the segment's directory>, "data": <JPEG bytes>}, which each model puts in its own form.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, Protocol

from ..files import write_atomic

# For fallbacks="default". The list form of fallbacks takes another beta, and each form 400s with the other's.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
IMAGE_TOKENS = (70, 140, 280, 560, 1120)  # the visual token budgets Gemma 4 takes per image
# No proxy, whatever http_proxy says: a request to this machine's server must not leave it.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Reply(NamedTuple):
    text: str
    model_id: str  # the model that wrote it: a server-side fallback can stand in for the one asked
    usage: dict


class ModelFailure(Exception):
    """A response with no answer in it, such as a refusal or a cut-off. Recorded, never parsed, with its details.

    A rerun asks again only if retry=True, as when the fallback couldn't run, the API turned the request away, or
    a local model's answer was cut off, unless a resend would repeat it.
    """

    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason, self.details = reason, details


class NotSent(Exception):
    """A dry run wrote the request instead of sending it."""


class VisionModel(Protocol):
    """What Stage 2 needs of a model: Claude, or an open-weight model on PACE."""

    model_id: str
    remote: bool  # whether requests leave this machine: the data guard holds remote models to synthetic data

    def respond(self, system: str, content: list[dict], schema: dict) -> Reply: ...


def _claude_block(block: dict, inline: bool) -> dict:
    if block["type"] != "image":
        return block
    data = base64.standard_b64encode(block["data"]).decode() if inline else f"ref:{block['ref']}"
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class ClaudeModel:
    """Claude through the Anthropic SDK, or with `dry_run`, a directory, the requests written there and not sent."""

    def __init__(self, model_id: str, effort: str, max_tokens: int, client=None, dry_run: Path | None = None):
        self.model_id, self.effort, self.max_tokens, self.client = model_id, effort, max_tokens, client
        self.dry_run, self.remote = dry_run, dry_run is None  # a dry run sends nothing, so it may size any segment

    def request(self, system: str, content: list[dict], schema: dict, inline_images: bool = True) -> dict:
        """The arguments for client.beta.messages.create. No thinking, sampling settings or prefill: those 400 here.

        Thinking is always on with this model, and the schema shapes the answer. Effort is explicit, as defaults differ.
        """
        return {
            "model": self.model_id, "max_tokens": self.max_tokens, "system": system,
            "messages": [{"role": "user", "content": [_claude_block(block, inline_images) for block in content]}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            "betas": [FALLBACK_BETA], "fallbacks": "default",  # a refusal reruns server-side on a recommended model
        }

    def respond(self, system: str, content: list[dict], schema: dict) -> Reply:
        if self.dry_run is not None:  # images by path; answers stored by a real run still serve a dry run
            text = json.dumps(self.request(system, content, schema, inline_images=False), indent=1) + "\n"
            write_atomic(self.dry_run / f"{hashlib.sha256(text.encode()).hexdigest()[:16]}.json", text.encode())
            raise NotSent
        import anthropic  # here, so the package works without the SDK, an optional extra
        self.client = self.client or anthropic.Anthropic()  # credentials from the environment or `ant auth login`
        try:
            response = self.client.beta.messages.create(**self.request(system, content, schema))
        except (anthropic.BadRequestError, anthropic.RequestTooLargeError) as error:
            # This request's fault, its size say, so the run goes on. The SDK has retried 429s and 5xx itself.
            raise ModelFailure(f"http_{error.status_code}", retry=True, message=error.message) from None
        usage = response.usage.model_dump(mode="json", exclude_none=True)  # iterations: every fallback attempt's
        # The stop reason before the content: a refusal or a cut-off can carry text that isn't an answer.
        if response.stop_reason != "end_turn":
            details = response.stop_details  # recommended_model is set when the fallback couldn't run
            raise ModelFailure(response.stop_reason, retry=bool(getattr(details, "recommended_model", None)),
                               category=getattr(details, "category", None), served_by=response.model, usage=usage)
        text = next((block.text for block in response.content if block.type == "text"), None)  # after any thinking
        if text is None:
            raise ModelFailure("no_text", served_by=response.model, usage=usage)
        return Reply(text, response.model, usage)


def _openai_block(block: dict) -> dict:
    if block["type"] != "image":
        return block
    data = base64.standard_b64encode(block["data"]).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


class LocalModel:
    """An open-weight model, such as Gemma 4, behind vLLM's OpenAI-compatible server on this machine."""

    remote = False  # the server shares this node, so frames sent to it never leave it

    def __init__(self, model_id: str, base_url: str, max_tokens: int, image_tokens: int,
                 temperature: float | None = None, timeout_s: float = 1800):
        if urllib.parse.urlsplit(base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("a local model's server must be on this machine")  # else frames would leave it
        if image_tokens not in IMAGE_TOKENS:
            raise ValueError(f"image_tokens must be one of {IMAGE_TOKENS}")
        self.model_id, self.max_tokens, self.image_tokens = model_id, max_tokens, image_tokens
        self.temperature = temperature  # None sends none, and vLLM samples with the model's own settings
        # An answer comes back whole, not streamed: 8,192 tokens at 20 a second take 7 minutes.
        self.url, self.timeout_s = base_url.rstrip("/") + "/chat/completions", timeout_s

    def request(self, system: str, content: list[dict], schema: dict) -> dict:
        """The body for /chat/completions: a system message, then the content as one user turn, images inline."""
        body = {
            "model": self.model_id, "max_completion_tokens": self.max_tokens,  # vLLM deprecates max_tokens
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [_openai_block(block) for block in content]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
            "mm_processor_kwargs": {"max_soft_tokens": self.image_tokens},  # vLLM's per-request image budget
        }
        if self.temperature is not None:  # only when set, so a config with none sends what it did before 2026-11-05
            body["temperature"] = self.temperature
        return body

    def respond(self, system: str, content: list[dict], schema: dict) -> Reply:
        body = json.dumps(self.request(system, content, schema)).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:  # a connection error goes up and stops the run: with no server, every request would meet it
            with _DIRECT.open(request, timeout=self.timeout_s) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as error:
            # A 404 says it serves another model: as with no server, every request would meet it, so the run stops.
            if error.code == 404 or not 400 <= error.code < 500:
                raise
            # This request's fault, its length say, so the run goes on. Its reason, cut short: it can echo the request.
            raise ModelFailure(f"http_{error.code}", retry=True,
                               message=error.read(1000).decode(errors="replace")) from None
        choice, usage = answer["choices"][0], answer.get("usage") or {}
        usage = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
        # The finish reason before the content: a cut-off carries text that isn't a whole answer.
        if choice["finish_reason"] != "stop":
            reason = "max_tokens" if choice["finish_reason"] == "length" else choice["finish_reason"]
            diagnosis, full = {}, reason == "max_tokens" and (usage["output_tokens"] or 0) >= self.max_tokens
            if full:
                # It used all its room. On 2026-11-04 Gemma looped on blank lines between JSON tokens: a cut-off
                # was 92% whitespace. The share and the end of its text say whether that's what happened.
                text = choice["message"].get("content") or ""
                diagnosis = {"whitespace_share": round(sum(map(str.isspace, text)) / max(len(text), 1), 3),
                             "tail": text[-300:]}
            # Asked again on a rerun while the model samples, since a resend usually finishes: 5 of 6 did on
            # 2026-11-04, at about 400 tokens. At temperature 0 a resend gives the same tokens, so a full-length one
            # would run away again on every rerun. A shorter one is asked again either way: vLLM 0.32 also ends an
            # answer where the context does, which a longer one may fit.
            retry = reason == "max_tokens" and not (full and self.temperature == 0)
            raise ModelFailure(reason, retry=retry, served_by=answer["model"], usage=usage, **diagnosis)
        if not choice["message"].get("content"):
            raise ModelFailure("no_text", served_by=answer["model"], usage=usage)
        return Reply(choice["message"]["content"], answer["model"], usage)


class FakeModel:
    """For tests: answers with `answer(system, content, schema)`, text or an exception to raise, and keeps requests."""

    def __init__(self, answer: Callable, model_id: str = "fake-model", remote: bool = False):
        self.answer, self.model_id, self.remote, self.requests = answer, model_id, remote, []

    def respond(self, system: str, content: list[dict], schema: dict) -> Reply:
        self.requests.append({"system": system, "content": content, "schema": schema})
        reply = self.answer(system, content, schema)
        if isinstance(reply, Exception):
            raise reply
        return Reply(reply, self.model_id, {})

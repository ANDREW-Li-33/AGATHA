"""Exercise the real Gemini / OpenAI SDK code paths against a local stub server (no network, no keys)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scenario_miner.data.synthetic import make_segment
from scenario_miner.prompts import build_request, parse_response
from scenario_miner.sample import save_sample
from scenario_miner.taxonomy import label_names, load_taxonomy

TAXONOMY = load_taxonomy()
LABELS = label_names(TAXONOMY)
ANSWER = {"labels": {n: {"present": n == "night", "confidence": 0.9 if n == "night" else 0.1} for n in LABELS},
          "notes": ""}


class _Stub(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        if "generateContent" in self.path:
            out = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(ANSWER)}]},
                                   "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 120,
                                     "thoughtsTokenCount": 30, "totalTokenCount": 1650}}
        else:
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": json.dumps(ANSWER)}}],
                   "usage": {"prompt_tokens": 900, "completion_tokens": 80, "total_tokens": 980}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def stub():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(scope="module")
def request_(tmp_path_factory):
    import numpy as np

    sample, jpeg = make_segment(np.random.default_rng(0), 0, n_samples=1)[0]
    save_sample(sample, jpeg, tmp_path_factory.mktemp("s"))
    return build_request(sample, "map_overlay_context", TAXONOMY, LABELS)


def test_gemini_backend(stub, request_, monkeypatch):
    pytest.importorskip("google.genai")
    from scenario_miner.vlm import GeminiBackend

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _Stub.requests.clear()
    res = GeminiBackend("gemini-2.5-flash", thinking_budget=0, base_url=stub).generate(request_, 0.0)
    assert parse_response(res.text, LABELS).decisions["night"] == (True, 0.9)
    assert (res.input_tokens, res.output_tokens) == (1500, 150)  # thinking tokens are billed as output

    path, body = _Stub.requests[-1]
    assert path.endswith("models/gemini-2.5-flash:generateContent")
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    schema = json.dumps(cfg.get("responseSchema") or cfg.get("responseJsonSchema"))
    assert "construction_zone" in schema and "confidence" in schema
    thinking = cfg["thinkingConfig"]
    assert thinking.get("thinkingBudget", thinking.get("thinking_budget")) == 0  # API accepts either casing
    parts = body["contents"][0]["parts"] if "parts" in body["contents"][0] else body["contents"]
    assert sum("inlineData" in p for p in parts) == 2  # raw + overlay images


def test_openai_compatible_backend(stub, request_):
    pytest.importorskip("openai")
    from scenario_miner.vlm import OpenAICompatBackend

    _Stub.requests.clear()
    res = OpenAICompatBackend("qwen2.5vl:7b", base_url=stub + "/v1").generate(request_, 0.0)
    assert parse_response(res.text, LABELS).error is None
    assert (res.input_tokens, res.output_tokens) == (900, 80)
    path, body = _Stub.requests[-1]
    assert path == "/v1/chat/completions" and body["response_format"] == {"type": "json_object"}
    content = body["messages"][0]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

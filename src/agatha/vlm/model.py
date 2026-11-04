"""The model interface, Claude behind it, which can also do a dry run that sends nothing, and a fake for tests.

Content blocks are model-neutral: text as Claude takes it, and images as {"type": "image", "ref": <path
under the segment's directory>, "data": <JPEG bytes>}, which each model puts in its own form.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, Protocol

from ..files import write_atomic

# For fallbacks="default". The list form of fallbacks takes another beta, and each form 400s with the other's.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Reply(NamedTuple):
    text: str
    model_id: str  # the model that wrote it: a server-side fallback can stand in for the one asked
    usage: dict


class ModelFailure(Exception):
    """A response with no answer in it, such as a refusal or a cut-off. Recorded, never parsed, with its details.

    A rerun asks again only if retry=True, as when the fallback couldn't run or the API turned the request away.
    """

    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason, self.details = reason, details


class NotSent(Exception):
    """A dry run wrote the request instead of sending it."""


class VisionModel(Protocol):
    """What Stage 2 needs of a model. An open-weight model on PACE can be one too."""

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

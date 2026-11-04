"""Prompt strategies, structured-output schema, and response parsing.

Strategies (the ablation axis of the experiment):
  zero_shot            raw image + label definitions
  cot                  zero_shot, but the model must list observations before deciding
  few_shot             zero_shot + K labeled examples from segments disjoint from eval
  map_overlay          raw image + the same frame with HD-map/3D-box overlay
  map_context          raw image + a text summary of the 3D scene (boxes, ranges, map)
  map_overlay_context  both of the above
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Union

from pydantic import BaseModel, create_model

from .features import object_features
from .render import OVERLAY_LEGEND, render_overlay, resize_max_side, to_jpeg
from .sample import Sample
from .taxonomy import Label

STRATEGIES = ("zero_shot", "cot", "few_shot", "map_overlay", "map_context", "map_overlay_context")


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/jpeg"


Part = Union[TextPart, ImagePart]


@dataclass
class VLMRequest:
    sample: Sample
    strategy: str
    labels: list[str]
    parts: list[Part]
    schema: type[BaseModel]
    cache_key: str = field(default="")


class LabelDecision(BaseModel):
    present: bool
    confidence: float


def response_model(labels: list[str], with_observations: bool) -> type[BaseModel]:
    label_model = create_model("Labels", **{name: (LabelDecision, ...) for name in labels})
    fields: dict = {}
    if with_observations:
        fields["observations"] = (list[str], ...)  # first, so the model writes them before deciding
    fields["labels"] = (label_model, ...)
    fields["notes"] = (str, "")
    return create_model("SceneLabels", **fields)


def _instructions(taxonomy: list[Label], labels: list[str], camera: str, cot: bool) -> str:
    defs = "\n".join(f"- {l.name}: {l.definition}" for l in taxonomy if l.name in labels)
    example = {"labels": {name: {"present": False, "confidence": 0.05} for name in labels}, "notes": ""}
    if cot:
        example = {"observations": ["..."], **example}
    steps = (
        "First write 3-8 short, factual observations in `observations` (road layout, objects and "
        "where they are, signage, lighting, weather). Then decide each label from those observations.\n"
        if cot else ""
    )
    return (
        f"You are labeling frames from an autonomous vehicle's {camera} camera for scenario mining.\n"
        "For EACH label below, decide whether it is present in this frame. `confidence` is your "
        "probability (0-1) that the label is present, so a confident 'no' is close to 0.\n\n"
        f"Labels:\n{defs}\n\n"
        "Rules: judge only what is visible (plus any 3D context provided). Do not guess from "
        "what is typical for this kind of road. Distances are approximate.\n"
        f"{steps}"
        "Reply with JSON only, in exactly this shape:\n"
        f"{json.dumps(example)}"
    )


def scene_context(sample: Sample, max_objects: int = 15) -> str:
    """Text summary of the 3D scene: what LiDAR boxes and the HD map say about this frame."""
    feats = sorted((f for f in object_features(sample) if f.in_camera), key=lambda f: f.range_m)
    counts = {t: sum(f.type == t for f in feats) for t in ("vehicle", "pedestrian", "cyclist", "sign")}
    lines = [
        "3D context from LiDAR boxes and the HD map (vehicle frame: x forward, y left, meters):",
        f"- Ego speed: {sample.ego_speed:.1f} m/s" if sample.ego_speed is not None else "- Ego speed: unknown",
        f"- Objects visible in this camera within 60 m: {counts['vehicle']} vehicles, "
        f"{counts['pedestrian']} pedestrians, {counts['cyclist']} cyclists, {counts['sign']} signs/static objects",
    ]
    for f in feats[:max_objects]:
        where = f"{f.x:.0f} m ahead, {abs(f.y):.0f} m {'left' if f.y >= 0 else 'right'}"
        tags = []
        if f.in_crosswalk:
            tags.append("IN CROSSWALK")
        elif f.lane_dist_m is not None and f.lane_dist_m <= 1.5:
            tags.append("ON A MAPPED TRAVEL LANE")
        if f.speed > 0.5:
            tags.append(f"moving {f.speed:.1f} m/s")
        size = f"{f.length:.1f}x{f.width:.1f}x{f.height:.1f} m"
        lines.append(f"  - {f.type} {size}: {where}" + (f" ({', '.join(tags)})" if tags else ""))
    kinds = [m.kind for m in sample.map_features]
    lines.append(
        f"- HD map within 60 m: {kinds.count('lane')} lane centerline pieces, "
        f"{kinds.count('crosswalk')} crosswalks, {kinds.count('stop_sign')} stop signs."
    )
    lines.append("Static objects on a mapped travel lane can mean the road differs from the map.")
    return "\n".join(lines)


def _image_part(sample: Sample, max_side: int, overlay: bool = False) -> ImagePart:
    img = sample.load_image()
    if overlay:
        img = render_overlay(sample, img)
    return ImagePart(to_jpeg(resize_max_side(img, max_side)))


def _answer_json(answer: dict[str, bool], labels: list[str]) -> str:
    return json.dumps({"labels": {n: {"present": bool(answer.get(n, False)),
                                      "confidence": 0.95 if answer.get(n) else 0.05} for n in labels},
                       "notes": ""})


def build_request(sample: Sample, strategy: str, taxonomy: list[Label], labels: list[str],
                  fewshot: list[tuple[Sample, dict[str, bool]]] = (), max_side: int = 1024) -> VLMRequest:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    cot = strategy == "cot"
    parts: list[Part] = [TextPart(_instructions(taxonomy, labels, sample.camera, cot))]

    if strategy == "few_shot":
        for i, (ex, answer) in enumerate(fewshot, 1):
            parts += [TextPart(f"Example {i}:"), _image_part(ex, max_side),
                      TextPart(f"Example {i} answer: {_answer_json(answer, labels)}")]
        parts.append(TextPart("Now label this frame:"))

    parts.append(_image_part(sample, max_side))
    if strategy in ("map_overlay", "map_overlay_context"):
        parts += [TextPart("The same frame with HD-map and 3D-box overlay. " + OVERLAY_LEGEND),
                  _image_part(sample, max_side, overlay=True)]
    if strategy in ("map_context", "map_overlay_context"):
        parts.append(TextPart(scene_context(sample)))

    req = VLMRequest(sample, strategy, labels, parts, response_model(labels, with_observations=cot))
    req.cache_key = request_hash(req)
    return req


def request_hash(req: VLMRequest) -> str:
    h = hashlib.sha256()
    for p in req.parts:
        h.update(b"T" + p.text.encode() if isinstance(p, TextPart) else b"I" + hashlib.sha256(p.data).digest())
    return h.hexdigest()


@dataclass
class ParsedResponse:
    decisions: dict[str, tuple[bool, float]]
    observations: list[str]
    error: str | None


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "present")
    return bool(v)


def parse_response(text: str, labels: list[str]) -> ParsedResponse:
    """Lenient parse: tolerates code fences, a flat {label: ...} dict, and bare booleans.

    A label the model left out counts as absent (confidence 0) and is reported in `error`.
    """
    try:
        obj = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return ParsedResponse({n: (False, 0.0) for n in labels}, [], f"unparseable: {e}")
    raw = obj.get("labels", obj) if isinstance(obj, dict) else {}
    decisions, missing = {}, []
    for name in labels:
        v = raw.get(name)
        if v is None:
            missing.append(name)
            decisions[name] = (False, 0.0)
        elif isinstance(v, dict):
            present = _as_bool(v.get("present", False))
            try:
                conf = float(v.get("confidence", 1.0 if present else 0.0))
            except (TypeError, ValueError):
                conf = 1.0 if present else 0.0
            decisions[name] = (present, min(1.0, max(0.0, conf)))
        else:
            present = _as_bool(v)
            decisions[name] = (present, 1.0 if present else 0.0)
    obs = obj.get("observations", []) if isinstance(obj, dict) else []
    error = f"missing labels: {', '.join(missing)}" if missing else None
    return ParsedResponse(decisions, [str(o) for o in obs] if isinstance(obs, list) else [], error)

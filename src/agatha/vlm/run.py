"""Pass 1, the merge, Pass 2, what they store under <segment directory>/stage2/, and the agatha-vlm CLI.

Raw answers, validation errors and accepted results are kept apart, as docs/stage2.md lists, and a request
answered before is answered from the store, so a rerun costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from pydantic import ValidationError

from .. import __version__
from ..config import load_config
from ..contracts import SCHEMA_VERSION, SceneWindow
from ..files import write_atomic
from ..loader import SegmentRef
from ..render import ensure_rendered
from ..report import _SYNTHETIC, context_label
from ..run_stage1 import _QuietParser
from ..scene_window import build_scene_window, default_windows, write_scene_window
from ..store import SegmentOutputs, load_segment, segment_directory
from .model import ClaudeModel, LocalModel, ModelFailure, NotSent, VisionModel
from .prompts import PASS1_SYSTEM, PASS2_SYSTEM, PROMPT_VERSION, request_content
from .schema import (
    CATEGORIES, PASS1_SCHEMA, PASS2_SCHEMA, TAXONOMY, Candidate, IncidentReport, Pass1Answer, Pass2Answer,
    SourceFrame, check_verdict, fit_candidate, fit_verdict,
)

# Like preprocessing_version: the package version, and a hash of this subpackage's source.
PIPELINE_VERSION = __version__ + "+" + hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()[:12]
_STEPS = {"pass1": (PASS1_SYSTEM, PASS1_SCHEMA, Pass1Answer), "pass2": (PASS2_SYSTEM, PASS2_SCHEMA, Pass2Answer)}


@dataclass(frozen=True)
class VlmConfig:
    model: dict  # the answering model's settings, [model]'s or [local]'s: a stored answer serves only the same ones
    margin_s: float
    fps: float
    retry_uncertain: bool
    retry_fps: float
    profile: str
    max_images: int  # per Pass 2 request, counting each camera's frames
    max_request_mb: float  # a Pass 2 request's images, as base64: the API turns away a request over 32 MB
    real_data_cleared: bool
    local: dict | None  # [local], for --model local: an open-weight model served on this machine


def load_vlm_config(path) -> VlmConfig:
    """configs/stage2.toml. Only a literal true clears real data: anything else, the string "true" too, fails closed.

    [local]'s temperature is optional. When set, it must be a number, 0 or more, and it's kept as a float, so 0, 0.0
    and -0.0 are one request. A ValueError says what's wrong.
    """
    cfg = tomllib.loads(Path(path).read_text())
    local = cfg.get("local")
    if local is not None and "temperature" in local:
        temperature = local["temperature"]
        number = isinstance(temperature, (int, float)) and not isinstance(temperature, bool)  # a bool is an int too
        if not (number and 0 <= temperature < math.inf):  # TOML has nan and inf, which aren't temperatures
            raise ValueError(f"[local] temperature must be a number, 0 or more, not {temperature!r}")
        # abs turns -0.0, which passes the check, into 0.0: the key's JSON writes -0.0, so it would key apart and
        # resend everything (found on 2026-11-05).
        local = {**local, "temperature": abs(float(temperature))}
    return VlmConfig(model=cfg["model"], **cfg["pass2"], real_data_cleared=cfg["data"]["real_data_cleared"] is True,
                     local=local)


class GuardRefused(Exception):
    """A remote model was given a segment that isn't synthetic, before real data was cleared."""


def check_guard(model: VisionModel, context_name: str, cleared: bool) -> None:
    """Fail closed: until real data is cleared, a remote model sees synthetic segments only. The tutorial's is real."""
    if model.remote and not cleared and not _SYNTHETIC.fullmatch(context_name):  # report.py's own test of it
        raise GuardRefused("a remote model runs only on synthetic segments until real_data_cleared")


def request_key(model_id: str, settings: dict, system: str, content: list[dict], schema: dict) -> str:
    """A hash of all that shapes an answer, so a stored answer serves only the same request. Images count by bytes."""
    blocks = [{"type": "image", "sha256": hashlib.sha256(block["data"]).hexdigest()} if block["type"] == "image"
              else block for block in content]
    text = json.dumps([model_id, settings, PROMPT_VERSION, system, blocks, schema], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _write_json(path: Path, value) -> None:
    write_atomic(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _same_event(a: dict, b: dict) -> bool:
    """Whether two candidates are one event: overlapping times, a shared category and a shared object, all three."""
    categories = {TAXONOMY[label][0] for label in a["labels"]} & {TAXONOMY[label][0] for label in b["labels"]}
    # Cones, barriers and debris have no boxes, so two candidates that name no object must share a label instead.
    shared = "object_aliases" if a["object_aliases"] or b["object_aliases"] else "labels"
    return (a["start_s"] <= b["end_s"] and b["start_s"] <= a["end_s"] and bool(set(a[shared]) & set(b[shared]))
            and bool(categories))


def merge_candidates(candidates: list[dict]) -> list[tuple[list[str], Candidate]]:
    """(source ids, merged candidate) per transitive group of duplicates. Candidates failing their checks sit out."""
    groups: list[list[dict]] = []
    for candidate in (item for item in candidates if not item["errors"]):
        # At most one candidate per window: the model split a window's own two apart, so they stay apart, even when
        # a neighboring window's candidate overlaps both.
        hits, windows = [], {candidate["window"]}
        for group in groups:
            owners = {item["window"] for item in group}
            if not owners & windows and any(_same_event(candidate, other) for other in group):
                hits, windows = [*hits, group], windows | owners
        groups = [group for group in groups if group not in hits] + [sum(hits, []) + [candidate]]  # hits as one
    merged = [([item["id"] for item in group], Candidate(
        labels=sorted({label for item in group for label in item["labels"]}, key=list(TAXONOMY).index),
        start_s=min(item["start_s"] for item in group), end_s=max(item["end_s"] for item in group),
        object_aliases=sorted({alias for item in group for alias in item["object_aliases"]}),
        evidence=list(dict.fromkeys(line for item in group for line in item["evidence"])),
        driving_impact="; ".join(dict.fromkeys(item["driving_impact"] for item in group)),
    )) for group in groups]
    return sorted(merged, key=lambda pair: (pair[1].start_s, pair[1].end_s, pair[0]))


def incident_report(answer: Pass2Answer, window: SceneWindow, model_id: str, incident_id: str) -> IncidentReport:
    """An accepted Pass 2 answer as an IncidentReport. check_verdict has found its aliases and frames already."""
    object_ids = {item.alias: item.laser_object_id for item in [*window.objects, *window.omitted_objects]}
    frames = {(frame.camera, round(frame.t_s, 2)): frame.frame_timestamp_micros for frame in window.frames}
    kept = {item.label for item in answer.label_assessments if item.status != "REJECTED"}
    return IncidentReport(
        schema_version=SCHEMA_VERSION, segment_id=window.segment_context_name, incident_id=incident_id,
        duplicate_of=None, start_s=answer.start_s, end_s=answer.end_s, verification_status=answer.status,
        categories=[category for category in CATEGORIES if any(TAXONOMY[label][0] == category for label in kept)],
        subcategories=[label for label in TAXONOMY if label in kept], label_assessments=answer.label_assessments,
        object_ids=list(dict.fromkeys(object_ids[alias] for alias in answer.object_aliases)),
        vlm_reasoning=answer.explanation, driving_impact=answer.driving_impact,
        uncertainty_reasons=answer.uncertainty_reasons,
        source_frames=[SourceFrame(camera=cited.camera, timestamp_s=round(cited.t_s, 2),
                                   frame_timestamp_micros=frames[cited.camera, round(cited.t_s, 2)])
                       for cited in answer.source_frames],
        pipeline_version=PIPELINE_VERSION, model_id=model_id, prompt_version=PROMPT_VERSION,
        preprocessing_version=window.preprocessing_version,
    )


def mark_duplicates(reports: list[IncidentReport]) -> None:
    """Set each report's duplicate_of to the first earlier report of the same event, if there is one.

    The same event: a label neither one rejected, an object, and overlapping times. Both reports stay, since Pass 2
    can find, in one candidate's window, the event another candidate was about.
    """
    for number, report in enumerate(reports):
        report.duplicate_of = next((earlier.incident_id for earlier in reports[:number]
                                    if set(earlier.subcategories) & set(report.subcategories)
                                    and set(earlier.object_ids) & set(report.object_ids)
                                    and earlier.start_s <= report.end_s and report.start_s <= earlier.end_s), None)


class Run:
    """One segment's two passes, with the store and the counts they share."""

    def __init__(self, segment: SegmentOutputs, config, vlm: VlmConfig, model: VisionModel):
        self.segment, self.config, self.vlm, self.model = segment, config, vlm, model
        self.directory, self.counts = segment.directory / "stage2", Counter()
        self.ref = SegmentRef(segment.meta["split"], segment.meta["context_name"])

    def _render(self, window: SceneWindow) -> SceneWindow:
        """The window, once every image it names is on disk. Images drawn before are kept."""
        ensure_rendered(window, self.segment, self.config.root, self.ref, self.config.profiles)
        return window

    def _ask(self, window: SceneWindow, hypothesis: Candidate | None = None) -> tuple:
        """Pass 1's request, or Pass 2's with a hypothesis: (outcome, key, window file, typed answer, stored record)."""
        check_guard(self.model, window.segment_context_name, self.vlm.real_data_cleared)  # however this was reached
        step = "pass1" if hypothesis is None else "pass2"
        system, schema, answer_type = _STEPS[step]
        window_file = write_scene_window(window, self.directory).name
        if not window.frames:  # nothing to show, so nothing to ask
            return "no_frames", None, window_file, None, None
        content = request_content(self._render(window), self.segment.directory, hypothesis)
        key = request_key(self.model.model_id, self.vlm.model, system, content, schema)
        path = self.directory / "responses" / f"{key}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if record is not None and not record.get("retry"):  # a stored failure too, unless asking again may work
            self.counts[f"{step}_reused"] += 1
        else:
            started = time.monotonic()
            try:
                reply = self.model.respond(system, content, schema)
            except NotSent:
                return "not_sent", key, window_file, None, None
            except ModelFailure as failure:
                outcome = {"failure": failure.reason, **failure.details}
            else:
                outcome = {"text": reply.text, "served_by": reply.model_id, "usage": reply.usage}
            self.counts[f"{step}_sent"] += 1
            record = {"step": step, "window": window_file, "model_id": self.model.model_id, "settings": self.vlm.model,
                      "prompt_version": PROMPT_VERSION, "latency_s": round(time.monotonic() - started, 3), **outcome}
            _write_json(path, record)
        if "failure" in record:
            return "failed", key, window_file, None, record
        try:
            return "answered", key, window_file, answer_type.model_validate_json(record["text"]), record
        except ValidationError as exc:
            self._errors(key, step, [f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()])
            return "invalid", key, window_file, None, record

    def _errors(self, key: str, step: str, errors: list[str]) -> None:
        """Write an answer's validation errors, or remove old ones that the checks no longer find."""
        path = self.directory / "errors" / f"{key}.json"
        if errors:
            _write_json(path, {"step": step, "errors": errors})
        else:
            path.unlink(missing_ok=True)

    def pass1(self, windows: list[SceneWindow]) -> tuple[list[dict], list[dict]]:
        """(what became of each window's request, every candidate), each candidate fitted to its window."""
        asked, candidates = [], []
        for number, window in enumerate(windows):
            outcome, key, window_file, answer, _ = self._ask(window)
            self.counts[f"pass1_{outcome}"] += 1
            asked.append({"id": f"W{number}", "window": window_file, "response": key, "outcome": outcome})
            if answer is None:
                continue
            found = [(f"W{number}-{index}", *fit_candidate(candidate, window))
                     for index, candidate in enumerate(answer.candidates)]
            self._errors(key, "pass1", [f"{id_}: {error}" for id_, _, _, errors in found for error in errors])
            candidates += [{"id": id_, "window": window_file, **fitted.model_dump(), "changes": changes,
                            "errors": errors} for id_, fitted, changes, errors in found]
        return asked, candidates

    def _window(self, hypothesis: Candidate, fps: float) -> SceneWindow:
        """Pass 2's window at `fps`, or at a lower rate that keeps it to max_images and, drawn, to max_request_mb."""
        last_t_s, margin_s = self.segment.meta["last_t_s"], self.vlm.margin_s
        start_s, end_s = max(hypothesis.start_s - margin_s, 0.0), min(hypothesis.end_s + margin_s, last_t_s)
        # A span of d s at r FPS holds d*r + 1 frames per camera, so r = (max_images / cameras - 1) / d keeps to it.
        fps = min(fps, (self.vlm.max_images / len(self.config.cameras) - 1) / max(end_s - start_s, 1e-3))
        build = partial(build_scene_window, self.segment, self.config, start_s, end_s, cameras=self.config.cameras,
                        profile=self.vlm.profile, closed_end=end_s >= last_t_s,
                        keep_aliases=hypothesis.object_aliases)  # its objects, whatever their rank
        window = self._render(build(fps))  # drawn, so its size is known: one too big is built again at a lower rate
        images = [self.segment.directory / frame.annotated_ref for frame in window.frames]
        size_mb = sum(path.stat().st_size for path in images) * 4 / 3e6  # as base64, which takes 4 bytes for every 3
        return window if size_mb <= self.vlm.max_request_mb else build(fps * self.vlm.max_request_mb / size_mb)

    def _verify(self, window: SceneWindow, hypothesis: Candidate) -> tuple[dict, tuple | None]:
        """(what became of one Pass 2 request, and if it was accepted (answer, window, the model that answered))."""
        outcome, key, window_file, answer, record = self._ask(window, hypothesis)
        accepted, changes = None, []
        if answer is not None:  # a stored answer too: it's fitted and checked again on every run
            answer, changes = fit_verdict(answer, window)  # a time between two frames shown cites the nearer
            errors = check_verdict(answer, window, hypothesis)
            self._errors(key, "pass2", errors)
            outcome = "invalid" if errors else answer.status
            accepted = None if errors else (answer, window, record["served_by"])
        self.counts[f"pass2_{outcome}"] += 1
        attempt = {"fps": round(window.requested_fps, 3), "window": window_file, "response": key, "outcome": outcome,
                   "changes": changes}
        return attempt, accepted

    def pass2(self, hypothesis: Candidate) -> tuple[list[dict], dict, tuple | None]:
        """(every attempt, the one whose outcome stands, its accepted answer). UNCERTAIN gets one finer retry."""
        window = self._window(hypothesis, self.vlm.fps)
        first, accepted = self._verify(window, hypothesis)
        attempts, stands = [first], first
        if first["outcome"] == "UNCERTAIN" and self.vlm.retry_uncertain:
            again = self._window(hypothesis, self.vlm.retry_fps)
            if len(again.frames) > len(window.frames):  # else the caps leave the model the same frames, or fewer
                self.counts["pass2_retried"] += 1
                retry, retried = self._verify(again, hypothesis)
                attempts.append(retry)
                if retried:  # a retry that fails leaves the first answer standing
                    stands, accepted = retry, retried
        return attempts, stands, accepted


def run_segment(segment: SegmentOutputs, config, vlm: VlmConfig, model: VisionModel) -> Counter:
    """Both passes over one segment, and their counts. A run that left requests unsent writes no results."""
    run = Run(segment, config, vlm, model)
    windows = default_windows(segment, config)
    asked, candidates = run.pass1(windows)
    merged, reports = [], []
    for number, (sources, hypothesis) in enumerate(merge_candidates(candidates), 1):
        incident_id = f"INC-{number:03d}"
        attempts, stands, accepted = run.pass2(hypothesis)  # response: the answer behind the outcome and any report
        merged.append({"incident_id": incident_id, "sources": sources, **hypothesis.model_dump(),
                       "outcome": stands["outcome"], "response": stands["response"], "attempts": attempts})
        if accepted:
            reports.append(incident_report(*accepted, incident_id))
    mark_duplicates(reports)
    counts = run.counts + Counter(item["outcome"] for item in merged)
    counts.update(windows=len(windows), candidates=len(candidates), merged=len(merged), reports=len(reports),
                  failing_checks=sum(bool(item["errors"]) for item in candidates),
                  duplicates=sum(report.duplicate_of is not None for report in reports),
                  unsent=counts["pass1_not_sent"] + counts["pass2_not_sent"])
    if not counts["unsent"]:
        _write_json(run.directory / "candidates.json", {"windows": asked, "candidates": candidates, "merged": merged})
        _write_json(run.directory / "incidents.json", [report.model_dump(mode="json") for report in reports])
    return counts


def _step(counts: Counter, step: str) -> str:
    return ", ".join(f"{counts[f'{step}_{what}']} {what.replace('_', ' ')}"
                     for what in ("sent", "reused", "not_sent", "no_frames", "failed", "invalid"))


def main(argv=None) -> int:
    os.umask(0o077)  # before anything is written: what's stored describes the segment
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    # A traceback prints paths, and they hold the context name. An exception's type names nothing.
    sys.excepthook = lambda kind, *_: print(f"agatha-vlm: failed ({kind.__name__})", file=sys.stderr)
    parser = _QuietParser(prog="agatha-vlm", description="Run Stage 2, the two-pass VLM, on one segment.")
    parser.add_argument("--config", default="configs/stage1.toml")
    parser.add_argument("--vlm-config", default="configs/stage2.toml")
    parser.add_argument("--split", default="training", help="the segment's split, which segment.json must match")
    parser.add_argument("--context", default="synthetic_cut_in")
    parser.add_argument("--model", choices=["local", "claude", "dry-run"], default="dry-run",
                        help="local asks the vLLM server in [local] on this machine, claude sends to Claude's API, and "
                             "dry-run writes each request to <segment>/stage2/dry_run/ and sends nothing")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.context):  # synthetic_x/../<name> would print <name> as a label
        parser.error("context")

    config = load_config(args.config)
    try:
        vlm = load_vlm_config(args.vlm_config)
    except ValueError as error:  # a negative temperature, say, or TOML it can't read: the message names no segment
        parser.exit(2, f"agatha-vlm: --vlm-config: {error}\n")
    directory = segment_directory(config, SegmentRef(args.split, args.context))
    if args.model == "local":
        if vlm.local is None:
            parser.error("local")
        local = vlm.local
        model = LocalModel(local["id"], local["base_url"], local["max_tokens"], local["image_tokens"],
                           local.get("temperature"))
        # Its answers key on its own settings, but not on base_url: where it's served doesn't change an answer. A
        # temperature is in the key when set, and a config with none keys as it did before 2026-11-05.
        vlm = replace(vlm, model={key: value for key, value in local.items() if key != "base_url"})
    else:
        model = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"],
                            dry_run=None if args.model == "claude" else directory / "stage2" / "dry_run")
    label = context_label(args.context)
    try:
        check_guard(model, args.context, vlm.real_data_cleared)  # before anything is read
        segment = load_segment(directory)
        if segment.meta["split"] != args.split:
            parser.error("split")
        shutil.rmtree(directory / "stage2" / "dry_run", ignore_errors=True)  # so it shows what this run left unsent
        counts = run_segment(segment, config, vlm, model)
    except GuardRefused:
        print(f"{label}: refused: a remote model runs on synthetic segments only", file=sys.stderr)
        return 1
    print(f"{label}: pass 1: {counts['windows']} windows: {_step(counts, 'pass1')}; {counts['candidates']} "
          f"candidates, {counts['failing_checks']} failing checks, {counts['merged']} after merging")
    print(f"{label}: pass 2: {_step(counts, 'pass2')}, {counts['pass2_retried']} retried; {counts['CONFIRMED']} "
          f"confirmed, {counts['REJECTED']} rejected, {counts['UNCERTAIN']} uncertain; incident reports: "
          + ("none written, as requests went unsent" if counts["unsent"]
             else f"{counts['reports']} written, {counts['duplicates']} marked duplicate"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

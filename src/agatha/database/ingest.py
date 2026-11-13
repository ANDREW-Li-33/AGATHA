"""Read Stage 1/2 outputs without changing them; build a validated, repeatable import.

Reports are immutable and identified by content, including model/prompt versions.
The same report in stage2/ and by_model/ is one incident. A changed report is a
new revision, so existing simulation jobs still point at the report they used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import TypeAdapter

from ..columns import CAMERA_NAMES, LIDAR_COMPONENTS, OBJECT_ID, STAGE1_COMPONENTS, TIME
from ..vlm.labels import HumanLabels
from ..vlm.schema import TAXONOMY, IncidentReport


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identity(kind: str, value) -> str:
    return str(uuid5(NAMESPACE_URL, f"agatha:{kind}:{digest(value)}"))


def read_json(path: Path):
    value = json.loads(path.read_bytes())
    canonical(value)  # reject NaN and Infinity, including inside otherwise untyped raw records
    return value


def source_refs(meta: dict) -> list[dict]:
    return [{"component": component, **{k: meta[k] for k in ("split", "context_name", "dataset_version")}}
            for component in (*STAGE1_COMPONENTS, *LIDAR_COMPONENTS)]


@dataclass
class ImportBundle:
    segment: dict
    run: dict
    incidents: list[dict]
    labels: list[dict]
    objects: list[dict]
    evidence: list[dict]
    human: dict | None


def _archive(directory: Path, candidates: dict, shared: Path) -> dict:
    """Keep original response text, usage, validation errors and request windows, even for failed attempts."""
    archive = {"candidates": candidates, "responses": {}, "errors": {}, "windows": {}, "missing_windows": []}
    attempts = [*candidates["windows"], *candidates["merged"],
                *(attempt for item in candidates["merged"] for attempt in item.get("attempts", []))]
    def locate(folder, name):
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid audit reference")
        return next((root / folder / name for root in (directory, shared) if (root / folder / name).is_file()), None)

    for key in sorted({item["response"] for item in attempts if item.get("response")}):
        path = locate("responses", f"{key}.json")
        if path is None:
            raise ValueError("candidates.json references a missing raw response")
        archive["responses"][path.name] = read_json(path)
        error = locate("errors", f"{key}.json")
        if error:
            archive["errors"][error.name] = read_json(error)
    windows = {item["window"] for item in [*attempts, *archive["responses"].values()] if item.get("window")}
    for name in sorted(windows):
        path = locate("windows", name)
        if path:
            archive["windows"][name] = read_json(path)
        else:
            archive["missing_windows"].append(name)  # older saved models may lack the request window cache
    return archive


def _validate_report(report: IncidentReport, meta: dict, frames: pd.DataFrame, tracks: pd.DataFrame) -> None:
    if report.schema_version != "0.1":
        raise ValueError("unsupported IncidentReport schema_version")
    if report.segment_id != meta["context_name"] or report.preprocessing_version != meta["preprocessing_version"]:
        raise ValueError("report and Stage 1 provenance disagree")
    # Stage 2 allows half a frame of jitter at a window edge. Keep its exact range.
    if not -0.05 <= report.start_s <= report.end_s <= meta["last_t_s"] + 0.05:
        raise ValueError("incident span is outside the segment")
    assessments = report.label_assessments
    assessed = [item.label for item in assessments]
    kept = {item.label for item in assessments if item.status != "REJECTED"}
    statuses = {item.status for item in assessments}
    expected = next((s for s in ("CONFIRMED", "UNCERTAIN") if s in statuses), "REJECTED")
    if not assessed or len(assessed) != len(set(assessed)):
        raise ValueError("label assessments must be nonempty and unique")
    if (set(report.subcategories) != kept or set(report.categories) != {TAXONOMY[s][0] for s in kept}
            or report.verification_status != expected):
        raise ValueError("incident status or retained labels disagree with assessments")
    if report.verification_status == "CONFIRMED" and not report.source_frames:
        raise ValueError("confirmed incident has no source frames")
    if report.verification_status == "UNCERTAIN" and not report.uncertainty_reasons:
        raise ValueError("uncertain incident has no uncertainty reasons")
    if len(set(report.object_ids)) != len(report.object_ids) or set(report.object_ids) - set(tracks[OBJECT_ID]):
        raise ValueError("report references duplicate or unknown object IDs")
    available = {(CAMERA_NAMES[int(row.camera)], int(getattr(row, TIME))): float(row.t_s)
                 for row in frames.itertuples()}
    for frame in report.source_frames:
        actual = available.get((frame.camera, frame.frame_timestamp_micros))
        if actual is None or not math.isfinite(frame.timestamp_s) or abs(actual - frame.timestamp_s) > 0.011:
            raise ValueError("source frame does not match its exact camera/timestamp key")
    for name in ("incident_id", "model_id", "prompt_version", "pipeline_version", "preprocessing_version"):
        if not getattr(report, name).strip():
            raise ValueError("report provenance fields must not be empty")


def prepare_import(segment_dir: Path, stage2_dir: Path | None = None, *, human_path: Path | None = None,
                   scene: str | None = None) -> ImportBundle:
    """Validate the entire scene before any database or Storage write."""
    segment_dir = Path(segment_dir)
    stage2_dir = Path(stage2_dir) if stage2_dir else segment_dir / "stage2"
    meta = read_json(segment_dir / "segment.json")
    if not re.fullmatch(r"[A-Za-z0-9_]+", meta["context_name"]):
        raise ValueError("invalid context name")
    if not math.isfinite(meta["last_t_s"]) or meta["last_t_s"] < 0:
        raise ValueError("invalid segment duration")
    segment_id = identity("segment", [meta[k] for k in
                                     ("context_name", "split", "dataset_version", "preprocessing_version")])
    segment = {"id": segment_id, **{k: meta[k] for k in
               ("context_name", "split", "dataset_version", "preprocessing_version", "t0_micros", "last_t_s")},
               "source_refs": source_refs(meta), "metadata": meta}
    raw = read_json(stage2_dir / "incidents.json")
    reports = TypeAdapter(list[IncidentReport]).validate_python(raw)
    frames = pd.read_parquet(segment_dir / "frames.parquet", columns=[TIME, "camera", "t_s"])
    tracks = pd.read_parquet(segment_dir / "tracks.parquet", columns=[TIME, OBJECT_ID, "type", "t_s"])
    ids, by_source = {}, {}
    for report, original in zip(reports, raw):
        _validate_report(report, meta, frames, tracks)
        if report.incident_id in by_source:
            raise ValueError("duplicate incident_id within one Stage 2 run")
        if report.duplicate_of is not None and report.duplicate_of not in ids:
            raise ValueError("duplicate_of must name an earlier report in the same run")
        ids[report.incident_id] = identity("incident", [segment_id, original, ids.get(report.duplicate_of)])
        by_source[report.incident_id] = report
    candidates = read_json(stage2_dir / "candidates.json")
    if not all(isinstance(candidates.get(k), list) for k in ("windows", "candidates", "merged")):
        raise ValueError("invalid candidates.json")
    merged = {item["incident_id"]: item for item in candidates["merged"]}
    if len(merged) != len(candidates["merged"]):
        raise ValueError("duplicate merged candidate")
    for report in reports:
        item = merged.get(report.incident_id)
        if item is None or item["outcome"] != report.verification_status:
            raise ValueError("report has no matching accepted candidate")
    if any(item["outcome"] in ("CONFIRMED", "UNCERTAIN", "REJECTED") and key not in ids
           for key, item in merged.items()):
        raise ValueError("accepted candidate has no report")
    archive = _archive(stage2_dir, candidates, segment_dir / "stage2")
    run_id = identity("run", [segment_id, raw, archive])
    run = {"id": run_id, "segment_id": segment_id, "archive": archive,
           "model_ids": sorted({r.model_id for r in reports} |
                               {r["model_id"] for r in archive["responses"].values() if r.get("model_id")}),
           "complete": bool(candidates["windows"]) and
                       all(w["outcome"] == "answered" for w in candidates["windows"]) and
                       all(not c.get("errors") for c in candidates["candidates"]) and
                       all(c["outcome"] in ("CONFIRMED", "UNCERTAIN", "REJECTED") for c in merged.values())}
    incidents, labels, objects, evidence = [], [], [], []
    for report, original in zip(reports, raw):
        iid = ids[report.incident_id]
        incidents.append({"id": iid, "segment_id": segment_id, "source_incident_id": report.incident_id,
                          "report_hash": digest([original, ids.get(report.duplicate_of)]), "raw_report": original,
                          "duplicate_of": ids.get(report.duplicate_of),
                          **{k: getattr(report, k) for k in ("start_s", "end_s", "categories", "subcategories",
                              "verification_status", "vlm_reasoning", "driving_impact", "pipeline_version",
                              "model_id", "prompt_version", "preprocessing_version")}})
        for assessment in report.label_assessments:
            labels.append({"incident_id": iid, "category": TAXONOMY[assessment.label][0], **assessment.model_dump()})
            for kind, lines in (("supporting", assessment.evidence), ("contradicting", assessment.contradicting_evidence)):
                for number, line in enumerate(lines):
                    evidence.append({"incident_id": iid, "evidence_key": f"{assessment.label}:{kind}:{number}",
                                     "kind": kind, "label": assessment.label, "observation": line})
        for number, frame in enumerate(report.source_frames):
            evidence.append({"incident_id": iid, "evidence_key": f"frame:{number}", "kind": "frame",
                             **frame.model_dump()})
        for oid in report.object_ids:
            whole = tracks[tracks[OBJECT_ID] == oid].sort_values(TIME)
            during = whole[whole.t_s.between(report.start_s - 0.05, report.end_s + 0.05)]
            # All observed classes are searchable; cyclist/pedestrian labels can switch mid-track.
            classes = sorted(set((during if not during.empty else whole)["type"]))
            objects.append({"incident_id": iid, "object_id": oid, "object_classes": classes})
    human = None
    if human_path is not None:
        if not scene:
            raise ValueError("human labels need the expected scene name")
        original = read_json(Path(human_path))
        parsed = HumanLabels.model_validate(original)
        if parsed.scene != scene or any(i.end_s > meta["last_t_s"] + 0.05 for i in parsed.incidents):
            raise ValueError("human labels name another scene or exceed its duration")
        human = {"id": identity("human", [segment_id, original]), "segment_id": segment_id, "scene": scene,
                 "annotation_version": digest(original), "reviewed": parsed.reviewed,
                 "no_incident": parsed.no_incident, "raw_labels": original}
    return ImportBundle(segment, run, incidents, labels, objects, evidence, human)


def result_directories(segment_dir: Path) -> list[Path]:
    """Top-level results and every saved model; content identities deduplicate copied results."""
    root = Path(segment_dir) / "stage2"
    return [p for p in [root, *sorted((root / "by_model").glob("*"))] if (p / "incidents.json").is_file()]

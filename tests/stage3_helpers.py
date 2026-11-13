"""Small synthetic Stage 1/2 files shared by Stage 3 tests; no Dataset access."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from agatha.columns import OBJECT_ID, TIME
from agatha.database.ingest import prepare_import


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def scene_files(root, *, context="synthetic_stage3", model="fake-model"):
    folder = root / context
    t0 = 1550000000000000
    times = [0, 100000, 200000, 400000, 500000, 800000, 1000000, 1500000]
    meta = {"schema_version": "0.1", "context_name": context, "split": "training", "dataset_version": "v2.0.1",
            "preprocessing_version": "synthetic-v1", "t0_micros": t0, "last_t_s": 1.5}
    write_json(folder / "segment.json", meta)
    frames = pd.DataFrame({TIME: [t0 + t for t in times], "camera": 1, "t_s": [t / 1e6 for t in times]})
    frames.to_parquet(folder / "frames.parquet", index=False)
    pd.DataFrame({TIME: frames[TIME], OBJECT_ID: "synthetic-car", "type": "vehicle", "t_s": frames.t_s}).to_parquet(
        folder / "tracks.parquet", index=False)
    render = folder / "render" / "FRONT"
    render.mkdir(parents=True, exist_ok=True)
    for i, timestamp in enumerate(frames[TIME]):
        image = np.full((48, 64, 3), (i * 25, 50, 100), np.uint8)
        (render / f"{timestamp}__clean.jpg").write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
        (render / f"{timestamp}__full.jpg").write_bytes(cv2.imencode(".jpg", 255 - image)[1].tobytes())
    reports = []
    for n, status in enumerate(("CONFIRMED", "REJECTED", "UNCERTAIN", "CONFIRMED"), 1):
        kept = status != "REJECTED"
        reports.append({"schema_version": "0.1", "segment_id": context, "incident_id": f"INC-{n:03}",
                        "duplicate_of": "INC-001" if n == 4 else None, "start_s": 0.4, "end_s": 0.8,
                        "categories": ["DYNAMIC_HAZARD"] if kept else [],
                        "subcategories": ["SUDDEN_CUT_IN"] if kept else [], "object_ids": ["synthetic-car"],
                        "verification_status": status,
                        "label_assessments": [{"label": "SUDDEN_CUT_IN", "status": status,
                                               "evidence": ["Moves across the path"],
                                               "contradicting_evidence": ["Clearance is hard to judge"]}],
                        "vlm_reasoning": "Synthetic explanation", "driving_impact": "Potential clearance reduction",
                        "uncertainty_reasons": ["Limited visibility"] if status == "UNCERTAIN" else [],
                        "source_frames": [{"camera": "FRONT", "timestamp_s": 0.4,
                                           "frame_timestamp_micros": t0 + 400000}],
                        "pipeline_version": "test-v1", "model_id": model, "prompt_version": "prompt-v1",
                        "preprocessing_version": meta["preprocessing_version"]})
    write_results(folder / "stage2", reports)
    return folder


def write_results(directory, reports):
    merged = [{"incident_id": r["incident_id"], "response": "pass2", "outcome": r["verification_status"],
               "labels": [a["label"] for a in r["label_assessments"]], "start_s": r["start_s"], "end_s": r["end_s"],
               "attempts": [{"response": "pass2", "outcome": r["verification_status"]}]} for r in reports]
    write_json(directory / "incidents.json", reports)
    write_json(directory / "candidates.json", {"windows": [{"response": "pass1", "outcome": "answered"}],
                                               "candidates": [], "merged": merged})
    for key in ("pass1", "pass2"):
        write_json(directory / "responses" / f"{key}.json",
                   {"text": "Original model response including whitespace\n", "model_id": "fake-model", "usage": {"input_tokens": 7}})


def bundle_at(tmp_path):
    return prepare_import(scene_files(tmp_path))

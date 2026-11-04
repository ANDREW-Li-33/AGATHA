"""DuckDB storage: schema, ingest of extracted samples, derived labels, splits."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb
import numpy as np

from .features import object_features
from .sample import iter_samples


def _sql(name: str) -> str:
    return resources.files("scenario_miner").joinpath("sql", name).read_text()


def connect(db_path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        con.execute(_sql("schema.sql"))
    return con


def ingest(con: duckdb.DuckDBPyConnection, extracted_root: str | Path) -> int:
    """Load extracted samples (idempotent), then recompute derived ground truth."""
    sample_rows, object_rows, gt_rows = [], [], []
    for s in iter_samples(extracted_root):
        ctx = s.context
        sample_rows.append((s.sample_id, s.segment_id, s.timestamp_us, s.camera, str(s.sample_dir.resolve()),
                            s.ego_speed, ctx.get("time_of_day"), ctx.get("weather"), ctx.get("location"),
                            ctx.get("source")))
        for f in object_features(s):
            object_rows.append((s.sample_id, f.object_id, f.type, f.x, f.y, f.z, f.length, f.width, f.height,
                                f.heading, f.speed, f.range_m, f.lane_dist_m, f.in_crosswalk, f.in_camera,
                                f.num_lidar_points))
        gt_rows += [(s.sample_id, label, bool(v), "synthetic") for label, v in s.gt.items()]
    if not sample_rows:
        return 0

    con.execute("BEGIN")
    con.execute("CREATE OR REPLACE TEMP TABLE _ids (sample_id VARCHAR)")
    con.executemany("INSERT INTO _ids VALUES (?)", [(r[0],) for r in sample_rows])
    con.execute("DELETE FROM objects WHERE sample_id IN (SELECT sample_id FROM _ids)")
    con.executemany("INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_rows)
    if object_rows:
        con.executemany("INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", object_rows)
    if gt_rows:
        con.executemany("INSERT OR REPLACE INTO ground_truth VALUES (?, ?, ?, ?)", gt_rows)
    con.execute(_sql("derived_labels.sql"))
    con.execute("COMMIT")
    return len(sample_rows)


def make_splits(con: duckdb.DuckDBPyConnection, fewshot_segments: int = 3, per_segment: int = 3,
                min_gap_s: float = 4.0, seed: int = 0) -> dict[str, int]:
    """Choose which samples get scored and which may be used as few-shot examples.

    * Few-shot examples come from whole segments that are never scored, so an
      example can't be a near-duplicate of a test frame 0.2 s later.
    * At most `per_segment` frames per eval segment, at least `min_gap_s` apart:
      consecutive frames are nearly identical, and a few long segments shouldn't
      dominate the metrics.
    """
    rows = con.execute("SELECT segment_id, sample_id, timestamp_us FROM samples ORDER BY segment_id, timestamp_us").fetchall()
    by_segment: dict[str, list[tuple[str, int]]] = {}
    for seg, sid, ts in rows:
        by_segment.setdefault(seg, []).append((sid, ts))

    rng = np.random.default_rng(seed)
    segments = sorted(by_segment)
    rng.shuffle(segments)
    assignments = []
    for i, seg in enumerate(segments):
        split = "fewshot" if i < fewshot_segments else "eval"
        frames = by_segment[seg]
        chosen: list[tuple[str, int]] = []
        for j in rng.permutation(len(frames)):
            sid, ts = frames[j]
            if all(abs(ts - t) >= min_gap_s * 1e6 for _, t in chosen):
                chosen.append((sid, ts))
            if len(chosen) == per_segment:
                break
        assignments += [(sid, split) for sid, _ in chosen]

    con.execute("BEGIN")
    con.execute("DELETE FROM splits")
    con.executemany("INSERT INTO splits VALUES (?, ?)", assignments)
    con.execute("COMMIT")
    counts: dict[str, int] = {}
    for _, split in assignments:
        counts[split] = counts.get(split, 0) + 1
    return counts

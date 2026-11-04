"""The frames table, segment.json, and reading and writing a segment's outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .columns import CAMERA_NAMES, TIME
from .contracts import SCHEMA_VERSION, SOURCE_DATASET_ATTRIBUTION
from .files import write_atomic
from .geometry import CameraModel, transform_from_list
from .loader import Poses, SegmentRef
from .tracks import pose_rows

_TABLES = ("frames", "ego", "tracks")


@dataclass(frozen=True)
class SegmentOutputs:
    directory: Path
    meta: dict             # segment.json
    frames: pd.DataFrame
    ego: pd.DataFrame
    tracks: pd.DataFrame


def build_frames_table(image_meta: pd.DataFrame, poses: Poses, gap_ms: float) -> pd.DataFrame:
    """One row per (frame, camera). Gaps are per camera, since a camera can lose a frame alone."""
    ts = image_meta[TIME].to_numpy()
    gap_before_ms = (image_meta.groupby("camera")[TIME].diff() / 1e3).to_numpy()
    pose_timestamp_s = image_meta["pose_timestamp_s"].to_numpy()
    return pd.DataFrame({
        TIME: ts,
        "t_s": (ts - poses.timestamps[0]) / 1e6,
        "camera": image_meta["camera"].to_numpy(),
        "camera_name": image_meta["camera"].map(CAMERA_NAMES).to_numpy(),
        "gap_before_ms": gap_before_ms,
        "has_gap_before": gap_before_ms > gap_ms,
        "pose_timestamp_s": pose_timestamp_s,
        "trigger_time_s": image_meta["trigger_time_s"].to_numpy(),
        "readout_done_time_s": image_meta["readout_done_time_s"].to_numpy(),
        "shutter_s": image_meta["shutter_s"].to_numpy(),
        "capture_offset_ms": pose_timestamp_s * 1e3 - ts / 1e3,
        "world_from_vehicle": poses.world_from_vehicle[pose_rows(poses, ts)].reshape(-1, 16).tolist(),
        "world_from_image_vehicle": [pose.tolist() for pose in image_meta["world_from_image_vehicle"]],
    })


def build_segment_meta(ref: SegmentRef, poses: Poses, cameras: dict[int, CameraModel], config) -> dict:
    t0_micros, last_micros = poses.timestamps[[0, -1]].tolist()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": config.dataset_version,
        "split": ref.split,
        "context_name": ref.context_name,
        "t0_micros": t0_micros,
        "last_t_s": (last_micros - t0_micros) / 1e6,
        "n_frames": len(poses.timestamps),
        "local_origin_world_m": poses.world_from_vehicle[0, :3, 3].tolist(),
        "rear_axle_to_front_m": config.rear_axle_to_front_m,
        "preprocessing_version": config.preprocessing_version,
        "attribution": SOURCE_DATASET_ATTRIBUTION,
        "cameras": {
            CAMERA_NAMES[camera_id]: {
                "camera_id": camera_id,
                "intrinsic": list(cam.intrinsic),
                "extrinsic": cam.extrinsic.reshape(-1).tolist(),
                "width": cam.width,
                "height": cam.height,
                "rolling_shutter_direction": cam.rolling_shutter_direction,
            }
            for camera_id, cam in cameras.items()
        },
    }


def write_segment(directory, meta: dict, frames: pd.DataFrame, ego: pd.DataFrame, tracks: pd.DataFrame) -> None:
    """segment.json goes first and comes back last: it's the commit point, so load_segment can't mix two runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "segment.json").unlink(missing_ok=True)
    for name, table in zip(_TABLES, (frames, ego, tracks)):
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), directory / f"{name}.parquet")
    text = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_atomic(directory / "segment.json", text.encode())


def _read_table(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()
    for field in table.schema:
        if pa.types.is_list(field.type):
            # to_pandas gives arrays, and `if cell:` or `==` on those don't behave like the lists written.
            df[field.name] = table[field.name].to_pylist()
    return df


def load_segment(directory) -> SegmentOutputs:
    directory = Path(directory)
    meta = json.loads((directory / "segment.json").read_text())
    return SegmentOutputs(directory, meta, *[_read_table(directory / f"{name}.parquet") for name in _TABLES])


def camera_models(meta: dict) -> dict[int, CameraModel]:
    """Calibration from segment.json, so later stages needn't reopen camera_calibration."""
    return {
        cam["camera_id"]: CameraModel(tuple(cam["intrinsic"]), transform_from_list(cam["extrinsic"]), cam["width"],
                                      cam["height"], cam["rolling_shutter_direction"])
        for cam in meta["cameras"].values()
    }


def segment_directory(config, ref: SegmentRef) -> Path:
    return config.outputs / ref.context_name

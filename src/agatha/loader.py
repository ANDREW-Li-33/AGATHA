"""Read Waymo Perception v2 Parquet components into sorted, tidy DataFrames.

Row order differs across components, so every read is sorted here and joins use
keys, never position. Each file holds one segment, so nothing filters on the
context name. Enums stay as Waymo's integer codes: `BOX_TYPES` names `type`, and
`CAMERA_NAMES` names `camera`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import (
    CAMERA, CAMERA_CALIBRATION, CAMERA_IMAGE, LIDAR_BOX, OBJECT, OBJECT_ID, PROJECTED_BOX, STAGE1_COMPONENTS,
    SYNCED_BOX, TIME, TIMESTAMP, VEHICLE_POSE, SchemaCheck, check_schema, col,
)
from .geometry import CameraModel, transform_from_list

_CAMERA = "camera"
_INTRINSICS = ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3")


@dataclass(frozen=True)
class SegmentRef:
    split: str
    context_name: str


class Poses(NamedTuple):
    timestamps: np.ndarray          # int64 (N,), sorted
    world_from_vehicle: np.ndarray  # float64 (N, 4, 4)


def component_path(root: Path, ref: SegmentRef, tag: str) -> Path:
    return root / ref.split / tag / f"{ref.context_name}.parquet"


def check_segment(root, ref) -> dict[str, SchemaCheck]:
    """Compare each file's schema against columns.py. Reads footers, never rows."""
    checks = {}
    for tag in STAGE1_COMPONENTS:
        path = component_path(root, ref, tag)
        checks[tag] = check_schema(tag, pq.read_schema(path)) if path.exists() else SchemaCheck(["<no file>"], [], 0)
    return checks


def _axis_columns(tag: str, path: str, name: str, axes: str = "xyz") -> dict[str, str]:
    """File column -> tidy name for each axis, e.g. box.size.x -> size_x."""
    return {col(tag, f"{path}.{a}"): name.format(a) for a in axes}


def _read(root, ref, tag: str, columns: dict[str, str], sort_by: list[str], camera_ids=None) -> pd.DataFrame:
    """Read `columns` (file name -> tidy name) in key order. An absent optional column loads as NaN."""
    path = component_path(root, ref, tag)
    absent = check_schema(tag, pq.read_schema(path)).missing_optional
    filters = None if camera_ids is None else [(CAMERA, "in", list(camera_ids))]
    table = pq.read_table(path, columns=[c for c in columns if c not in absent], filters=filters)
    # ignore_metadata: pandas metadata could move a key into the index, and reindex would then fill it with NaN.
    df = table.to_pandas(ignore_metadata=True).rename(columns=columns).reindex(columns=list(columns.values()))
    # A dictionary column arrives as a Categorical, which sorts by the file's dictionary, and that would move aliases.
    decoded = df.astype({name: df[name].cat.categories.dtype for name in df.select_dtypes("category")})
    return decoded.sort_values(sort_by, ignore_index=True)


def load_poses(root, ref) -> Poses:
    columns = {TIMESTAMP: TIME, col(VEHICLE_POSE, "world_from_vehicle.transform"): "world_from_vehicle"}
    df = _read(root, ref, VEHICLE_POSE, columns, [TIME])
    return Poses(df[TIME].to_numpy(), np.stack([transform_from_list(v) for v in df["world_from_vehicle"]]))


def load_cameras(root, ref, camera_ids) -> dict[int, CameraModel]:
    """Calibration by camera id. It has no timestamp: one row per camera per segment."""
    columns = {
        CAMERA: _CAMERA,
        **{col(CAMERA_CALIBRATION, f"intrinsic.{k}"): k for k in _INTRINSICS},
        col(CAMERA_CALIBRATION, "extrinsic.transform"): "extrinsic",
        col(CAMERA_CALIBRATION, "width"): "width",
        col(CAMERA_CALIBRATION, "height"): "height",
        col(CAMERA_CALIBRATION, "rolling_shutter_direction"): "rolling_shutter_direction",
    }
    df = _read(root, ref, CAMERA_CALIBRATION, columns, [_CAMERA], camera_ids)
    return {
        int(row.camera): CameraModel(
            intrinsic=tuple(float(getattr(row, k)) for k in _INTRINSICS),
            extrinsic=transform_from_list(row.extrinsic),
            width=int(row.width),
            height=int(row.height),
            rolling_shutter_direction=int(row.rolling_shutter_direction),
        )
        for row in df.itertuples()
    }


def load_image_metadata(root, ref, camera_ids) -> pd.DataFrame:
    """One row per (frame, camera). Never reads the image column, which is most of the file."""
    linear = _axis_columns(CAMERA_IMAGE, "velocity.linear_velocity", "linear_velocity_{}_mps")
    columns = {
        TIMESTAMP: TIME,
        CAMERA: _CAMERA,
        col(CAMERA_IMAGE, "pose_timestamp"): "pose_timestamp_s",
        col(CAMERA_IMAGE, "rolling_shutter_params.camera_trigger_time"): "trigger_time_s",
        col(CAMERA_IMAGE, "rolling_shutter_params.camera_readout_done_time"): "readout_done_time_s",
        col(CAMERA_IMAGE, "rolling_shutter_params.shutter"): "shutter_s",
        col(CAMERA_IMAGE, "pose.transform"): "world_from_image_vehicle",
        **linear,
    }
    df = _read(root, ref, CAMERA_IMAGE, columns, [TIME, _CAMERA], camera_ids)
    # The files hold linear_velocity as float32, and everything it meets downstream is float64.
    return df.astype({name: np.float64 for name in linear.values()})


def load_lidar_boxes(root, ref) -> pd.DataFrame:
    optional = {**_axis_columns(LIDAR_BOX, "speed", "speed_{}"), **_axis_columns(LIDAR_BOX, "acceleration", "accel_{}")}
    columns = {
        TIMESTAMP: TIME,
        OBJECT: OBJECT_ID,
        **_axis_columns(LIDAR_BOX, "box.center", "center_{}"),
        **_axis_columns(LIDAR_BOX, "box.size", "size_{}"),
        col(LIDAR_BOX, "box.heading"): "heading",
        col(LIDAR_BOX, "type"): "type",
        col(LIDAR_BOX, "num_lidar_points_in_box"): "num_lidar_points",
        col(LIDAR_BOX, "num_top_lidar_points_in_box"): "num_top_lidar_points",
        **optional,
    }
    df = _read(root, ref, LIDAR_BOX, columns, [TIME, OBJECT_ID])
    # A column with no values can be Arrow's null type, which reaches pandas as objects holding None.
    return df.astype({name: np.float64 for name in optional.values()})


def load_synced_boxes(root, ref) -> pd.DataFrame:
    columns = {
        TIMESTAMP: TIME,
        OBJECT: OBJECT_ID,
        col(SYNCED_BOX, "most_visible_camera_name"): "most_visible_camera",
        **_axis_columns(SYNCED_BOX, "camera_synced_box.center", "center_{}"),
        **_axis_columns(SYNCED_BOX, "camera_synced_box.size", "size_{}"),
        col(SYNCED_BOX, "camera_synced_box.heading"): "heading",
    }
    return _read(root, ref, SYNCED_BOX, columns, [TIME, OBJECT_ID])


def load_projected_boxes(root, ref, camera_ids) -> pd.DataFrame:
    columns = {
        TIMESTAMP: TIME,
        CAMERA: _CAMERA,
        OBJECT: OBJECT_ID,
        **_axis_columns(PROJECTED_BOX, "box.center", "center_{}", "xy"),
        **_axis_columns(PROJECTED_BOX, "box.size", "size_{}", "xy"),
        col(PROJECTED_BOX, "type"): "type",
    }
    return _read(root, ref, PROJECTED_BOX, columns, [TIME, _CAMERA, OBJECT_ID], camera_ids)


def read_jpegs(root, ref, camera_id, timestamps=None) -> dict[int, bytes]:
    """JPEG bytes by timestamp. pyarrow applies the filters, so skipped images never reach Python."""
    filters = [(CAMERA, "=", camera_id)]
    if timestamps is not None:  # an empty selection means no images, not all of them
        filters.append((TIMESTAMP, "in", list(timestamps)))
    image = col(CAMERA_IMAGE, "image")
    table = pq.read_table(component_path(root, ref, CAMERA_IMAGE), columns=[TIMESTAMP, image], filters=filters)
    return dict(zip(table[TIMESTAMP].to_pylist(), table[image].to_pylist()))

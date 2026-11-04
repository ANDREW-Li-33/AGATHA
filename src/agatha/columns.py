"""Waymo Perception v2 component tags, Parquet column names, and enums.

Columns are named key.<field> and [<ClassName>].<dotted.path>, so plain PyArrow
can read everything. The waymo-open-dataset package isn't used: it pins
TensorFlow and no longer installs from PyPI (its jaxlib pin is gone).

These lists were derived by hand from the dataclasses in
src/waymo_open_dataset/v2/, not read from a real file, so `check_schema` compares
them against every file before anything is loaded.
"""

from __future__ import annotations

from typing import NamedTuple

import pyarrow as pa

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
CAMERA_IDS = {name: i for i, name in CAMERA_NAMES.items()}
BOX_TYPES = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
ROLLING_SHUTTER = {0: "UNKNOWN", 1: "TOP_TO_BOTTOM", 2: "LEFT_TO_RIGHT", 3: "BOTTOM_TO_TOP",
                   4: "RIGHT_TO_LEFT", 5: "GLOBAL_SHUTTER"}

SEGMENT = "key.segment_context_name"
TIMESTAMP = "key.frame_timestamp_micros"
CAMERA = "key.camera_name"
OBJECT = "key.laser_object_id"

CAMERA_IMAGE = "camera_image"
CAMERA_CALIBRATION = "camera_calibration"
VEHICLE_POSE = "vehicle_pose"
LIDAR_BOX = "lidar_box"
SYNCED_BOX = "lidar_camera_synced_box"
PROJECTED_BOX = "projected_lidar_box"
STATS = "stats"  # the cut-in scan's, beside lidar_box and vehicle_pose. Stage 1 never reads it.

STAGE1_COMPONENTS = (CAMERA_IMAGE, CAMERA_CALIBRATION, VEHICLE_POSE, LIDAR_BOX, SYNCED_BOX, PROJECTED_BOX)
# Named in every SceneWindow for Stage 4. Nothing here reads them.
LIDAR_COMPONENTS = ("lidar", "lidar_calibration", "lidar_pose")

# The keys' tidy names, in every table from the loader on.
TIME = "frame_timestamp_micros"
OBJECT_ID = "laser_object_id"

_CLASSES = {
    CAMERA_IMAGE: "CameraImageComponent",
    CAMERA_CALIBRATION: "CameraCalibrationComponent",
    VEHICLE_POSE: "VehiclePoseComponent",
    LIDAR_BOX: "LiDARBoxComponent",
    SYNCED_BOX: "LiDARCameraSyncedBoxComponent",
    PROJECTED_BOX: "ProjectedLiDARBoxComponent",
    STATS: "StatsComponent",
}


def col(tag: str, path: str) -> str:
    """Full column name, e.g. col("lidar_box", "box.center.x")."""
    return f"[{_CLASSES[tag]}].{path}"


def _xyz(tag: str, path: str) -> list[str]:
    return [col(tag, f"{path}.{axis}") for axis in "xyz"]


# Everything a Stage 1 run or the cut-in scan needs, keyed by component. A file missing
# any of these fails the schema check. Extra columns are fine.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    CAMERA_IMAGE: [
        SEGMENT, TIMESTAMP, CAMERA,
        col(CAMERA_IMAGE, "image"),
        col(CAMERA_IMAGE, "pose.transform"),
        *_xyz(CAMERA_IMAGE, "velocity.linear_velocity"),
        col(CAMERA_IMAGE, "pose_timestamp"),
        col(CAMERA_IMAGE, "rolling_shutter_params.shutter"),
        col(CAMERA_IMAGE, "rolling_shutter_params.camera_trigger_time"),
        col(CAMERA_IMAGE, "rolling_shutter_params.camera_readout_done_time"),
    ],
    CAMERA_CALIBRATION: [
        SEGMENT, CAMERA,
        *[col(CAMERA_CALIBRATION, f"intrinsic.{k}")
          for k in ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3")],
        col(CAMERA_CALIBRATION, "extrinsic.transform"),
        col(CAMERA_CALIBRATION, "width"),
        col(CAMERA_CALIBRATION, "height"),
        col(CAMERA_CALIBRATION, "rolling_shutter_direction"),
    ],
    VEHICLE_POSE: [SEGMENT, TIMESTAMP, col(VEHICLE_POSE, "world_from_vehicle.transform")],
    LIDAR_BOX: [
        SEGMENT, TIMESTAMP, OBJECT,
        *_xyz(LIDAR_BOX, "box.center"),
        *_xyz(LIDAR_BOX, "box.size"),
        col(LIDAR_BOX, "box.heading"),
        col(LIDAR_BOX, "type"),
        col(LIDAR_BOX, "num_lidar_points_in_box"),
        col(LIDAR_BOX, "num_top_lidar_points_in_box"),
    ],
    SYNCED_BOX: [
        SEGMENT, TIMESTAMP, OBJECT,
        col(SYNCED_BOX, "most_visible_camera_name"),
        *_xyz(SYNCED_BOX, "camera_synced_box.center"),
        *_xyz(SYNCED_BOX, "camera_synced_box.size"),
        col(SYNCED_BOX, "camera_synced_box.heading"),
    ],
    PROJECTED_BOX: [
        SEGMENT, TIMESTAMP, CAMERA, OBJECT,
        col(PROJECTED_BOX, "box.center.x"),
        col(PROJECTED_BOX, "box.center.y"),
        col(PROJECTED_BOX, "box.size.x"),
        col(PROJECTED_BOX, "box.size.y"),
        col(PROJECTED_BOX, "type"),
    ],
    # One row per frame, but each value holds for the whole segment. The object counts aren't read.
    STATS: [SEGMENT, TIMESTAMP, col(STATS, "time_of_day"), col(STATS, "location"), col(STATS, "weather")],
}

# Declared Optional in the v2 dataclasses, so the column can be null or, possibly,
# absent. Loaders treat an absent column as all null.
OPTIONAL_COLUMNS: dict[str, list[str]] = {
    LIDAR_BOX: [*_xyz(LIDAR_BOX, "speed"), *_xyz(LIDAR_BOX, "acceleration")],
}


class SchemaCheck(NamedTuple):
    missing: list[str]           # required, and absent: the file can't be used
    missing_optional: list[str]  # optional, and absent: loaded as null
    extra: int                   # columns we don't use

    @property
    def ok(self) -> bool:
        return not self.missing


def check_schema(tag: str, schema: pa.Schema) -> SchemaCheck:
    names = set(schema.names)
    required = REQUIRED_COLUMNS[tag]
    optional = OPTIONAL_COLUMNS.get(tag, [])
    return SchemaCheck(
        missing=[c for c in required if c not in names],
        missing_optional=[c for c in optional if c not in names],
        extra=len(names - set(required) - set(optional)),
    )

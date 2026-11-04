"""A synthetic segment in the real Waymo v2 Parquet layout, for runs with no Dataset access.

The scene is analytic: an ego and nine objects on a curved road, in a world frame with a
remote origin, pushed through the same transforms as real data. So a transform bug shows
up as residuals.

`projected_lidar_box` is the oracle, and here it comes from the same projection that
checks.py compares it against. Label residuals are near zero by construction, and that
proves only the wiring, not that the projection matches Waymo's. tests/test_geometry.py does
that, against Waymo's own vectors. This oracle has no object motion, unlike Waymo's, so
synced boxes on moving objects read up to about 3 px, the reverse of real data.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .columns import (
    BOX_TYPES, CAMERA_CALIBRATION, CAMERA_IDS, CAMERA_IMAGE, LIDAR_BOX, OPTIONAL_COLUMNS, PROJECTED_BOX,
    REQUIRED_COLUMNS, STAGE1_COMPONENTS, SYNCED_BOX, VEHICLE_POSE,
)
from .geometry import (
    CameraModel, box_corners, camera_from_label, inv_se3, project_camera_points, rot_z, se3,
    transform_from_list, transform_points, wireframe, wrap_angle, yaw_of,
)
from .loader import SegmentRef, component_path

# FRONT's calibration comes from Waymo's camera_model_test.cc, see NOTICE.
_INTRINSIC = (2055.55614936, 2055.55614936, 939.657469886, 641.072182194, 0.032316008498,
              -0.321412482553, 0.000793258395371, -0.000625749354133, 0.0)
_FRONT_EXTRINSIC = transform_from_list([
    0.999892684989, -0.00599320840002, 0.0133678704017, 1.53891424471,
    0.00604223652133, 0.999975156055, -0.0036302411765, -0.0236339408393,
    -0.0133457814992, 0.00371062343188, 0.999904056092, 2.11527057298,
    0, 0, 0, 1,
])
_WIDTH, _HEIGHT = 1920, 1280
_LEFT_TO_RIGHT = 2
_CAMERA_YAW_RAD = {"FRONT": 0.0, "FRONT_LEFT": np.pi / 4, "FRONT_RIGHT": -np.pi / 4}

# Seconds after frame_timestamp_micros. Each image's pose is at its capture time. Labels and
# vehicle_pose sit near FRONT's time but not on it, so skipping camera_from_label is a real error.
IMAGE_OFFSET_S = {"FRONT_LEFT": 0.0375, "FRONT": 0.05, "FRONT_RIGHT": 0.0625}
LABEL_OFFSET_S = 0.048

_FIRST_FRAME_MICROS = 1_550_000_000_000_000
_READOUT_S = 0.044     # from one real FRONT image
_POSE_LAG_S = 0.0043   # the same image's pose, after the middle of its exposure
_NULL_LABEL_RATE = 0.03

_ROAD_CENTER = np.array([-4000.0, 11500.0, 86.5])  # the world origin is remote
_ROAD_RADIUS_M = 200.0
_ROAD_START_YAW_RAD = 2.0
_LANE_M = 3.7
_LABELED_RANGE_M = 75.0

_SKY_BGR, _ROAD_BGR = (235, 215, 190), (110, 110, 110)
_BOX_BGR = {"vehicle": (170, 110, 50), "pedestrian": (60, 60, 220), "sign": (60, 200, 230)}
_TOP_FACE = [0, 1, 5, 4]  # in box_corners order
_BOX_TYPE_IDS = {name: i for i, name in BOX_TYPES.items()}

# The real types, from Waymo's dataclasses, by the last part of the column name. Anything else
# is float64, except linear_velocity, which is float32 in the real files and a trap for readers.
_ARROW_TYPES = {
    "segment_context_name": pa.string(), "laser_object_id": pa.string(),
    "frame_timestamp_micros": pa.int64(), "num_lidar_points_in_box": pa.int64(),
    "num_top_lidar_points_in_box": pa.int64(), "camera_name": pa.int8(), "type": pa.int8(),
    "most_visible_camera_name": pa.int8(), "rolling_shutter_direction": pa.int8(),
    "width": pa.int32(), "height": pa.int32(), "image": pa.binary(),
    "transform": pa.list_(pa.float64(), 16),
}


def _road_yaw(s: float) -> float:
    return _ROAD_START_YAW_RAD + s / _ROAD_RADIUS_M


def _road_point(s: float, d: float, z: float = 0.0) -> np.ndarray:
    """World position s meters along the left-hand curve, d meters left of it."""
    yaw = _road_yaw(s)
    radius = _ROAD_RADIUS_M - d
    return _ROAD_CENTER + np.array([radius * np.sin(yaw), -radius * np.cos(yaw), z])


def _ego_s(t: float) -> float:
    """14 m/s falling to 6 over 20 s. On this curve, yaw rate runs from 0.07 to 0.03 rad/s.

    The yaw is the point: it's what makes differencing vehicle-frame positions invent lateral
    motion, as it does on real data.
    """
    return 14.0 * t - 0.2 * t * t


def _ego_position(t: float) -> np.ndarray:
    return _road_point(_ego_s(t), 0.0)


def _ego_pose(t: float) -> np.ndarray:
    return se3(rot_z(_road_yaw(_ego_s(t))), _ego_position(t))


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class _Actor(NamedTuple):
    box_type: str
    size: tuple[float, float, float]               # length, width, height
    s: Callable[[float], float]                    # meters along the curve at time t
    d: Callable[[float], float]                    # meters left of the curve at time t
    yaw_rad: float = 0.0                           # heading relative to the curve
    bottom_m: float = 0.0                          # box bottom above the road
    no_points_s: tuple[float, float] = (0.0, 0.0)  # when LiDAR gets no points in the box

    def position(self, t: float) -> np.ndarray:
        """World position of the box center."""
        return _road_point(self.s(t), self.d(t), self.bottom_m + self.size[2] / 2.0)


_VEHICLE = (4.6, 1.9, 1.6)

_ACTORS = {
    # Overtakes on the right, unseen by FRONT at first, then enters the ego path from 8 to 12 s.
    "cut_in": _Actor("vehicle", _VEHICLE, lambda t: _ego_s(t) + 3.0 + 1.6 * t,
                     lambda t: -_LANE_M * (1.0 - _smoothstep((t - 8.0) / 4.0))),
    "lead": _Actor("vehicle", _VEHICLE, lambda t: _ego_s(t) + 50.0 - 0.2 * t, lambda t: 0.0),
    "parked": _Actor("vehicle", _VEHICLE, lambda t: 90.0, lambda t: -6.0),
    # Holds station in the next lane: the ghost-box case for FRONT.
    "beside": _Actor("vehicle", _VEHICLE, lambda t: _ego_s(t) + 4.5, lambda t: _LANE_M),
    # Waits on the left until the traffic has passed, then crosses just ahead of the ego.
    "pedestrian": _Actor("pedestrian", (0.6, 0.6, 1.75), lambda t: 205.0,
                         lambda t: float(np.clip(1.4 * (19.0 - t), -7.0, 7.0)), yaw_rad=-np.pi / 2),
    "sign": _Actor("sign", (0.1, 0.8, 0.8), lambda t: 150.0, lambda t: -6.5, yaw_rad=np.pi, bottom_m=1.5),
    # Follows the cut-in down the right lane, hidden behind it for part of the way.
    "occluded": _Actor("vehicle", _VEHICLE, lambda t: _ego_s(t) + 35.0 - 0.5 * t, lambda t: -_LANE_M,
                       no_points_s=(5.0, 9.0)),
    # Well off the road. Anything nearer is clamped by the time a side camera sees it, which
    # would leave that camera with no boxes for check 7 to judge.
    "parked_left": _Actor("vehicle", _VEHICLE, lambda t: 120.0, lambda t: 11.0),
    "parked_right": _Actor("vehicle", _VEHICLE, lambda t: 160.0, lambda t: -11.0),
}

# Shaped like real 22-character base64url IDs, but readable, so a test can pick out one object.
OBJECT_IDS = {name: name.replace("_", "-").upper().ljust(22, "0") for name in _ACTORS}


class _Label(NamedTuple):
    name: str
    actor: _Actor
    center: np.ndarray  # in the vehicle_pose frame
    heading: float


class _Seen(NamedTuple):
    rectangle: tuple[float, float, float, float]  # u_min, u_max, v_min, v_max, clamped to the image
    corners: np.ndarray                           # (8, 3) in the camera frame

    @property
    def area(self) -> float:
        u_min, u_max, v_min, v_max = self.rectangle
        return (u_max - u_min) * (v_max - v_min)


def _motion(position: Callable[[float], np.ndarray], t: float, h: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """World velocity and acceleration, by differences far finer than any frame step."""
    before, now, after = position(t - h), position(t), position(t + h)
    return (after - before) / (2.0 * h), (after - 2.0 * now + before) / (h * h)


def _camera_model(name: str) -> CameraModel:
    extrinsic = se3(rot_z(_CAMERA_YAW_RAD[name]), np.zeros(3)) @ _FRONT_EXTRINSIC
    return CameraModel(_INTRINSIC, extrinsic, _WIDTH, _HEIGHT, _LEFT_TO_RIGHT)


def _frame_timestamps(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """10 Hz with sub-ms jitter. Two ticks are dropped, so n_frames is what's left."""
    ticks = np.delete(np.arange(n_frames + 2), [n_frames // 3, 2 * n_frames // 3])
    return _FIRST_FRAME_MICROS + ticks * 100_000 + rng.integers(-400, 400, ticks.size)


def _labels(t: float, world_from_vehicle: np.ndarray) -> list[_Label]:
    """Each actor within the labeled range at time t, in that frame's vehicle_pose frame."""
    vehicle_from_world = inv_se3(world_from_vehicle)
    labels = []
    for name, actor in _ACTORS.items():
        center = transform_points(vehicle_from_world, actor.position(t))[0]
        if np.hypot(center[0], center[1]) <= _LABELED_RANGE_M:
            heading = wrap_angle(_road_yaw(actor.s(t)) + actor.yaw_rad - yaw_of(world_from_vehicle))
            labels.append(_Label(name, actor, center, heading))
    return labels


def _clamped_rectangle(corners_camera: np.ndarray, cam: CameraModel) -> tuple[float, float, float, float] | None:
    """Waymo's rule: the corners' rectangle, clamped to the image, or None if fully outside.

    Corners the lens model rejects are skipped. How Waymo treats them is undocumented, and
    checks.py leaves such boxes out anyway.
    """
    proj = project_camera_points(corners_camera, cam)
    if not proj.valid.any():
        return None
    limits = [cam.width, cam.height]
    low = np.clip(proj.uv[proj.valid].min(axis=0), 0.0, limits)
    high = np.clip(proj.uv[proj.valid].max(axis=0), 0.0, limits)
    if (low == high).any():
        return None
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def _view(labels: list[_Label], camera_from_frame: np.ndarray, cam: CameraModel) -> dict[str, _Seen]:
    """The boxes in one image, by object name."""
    view = {}
    for label in labels:
        corners = transform_points(camera_from_frame, box_corners(label.center, label.actor.size, label.heading))
        rectangle = _clamped_rectangle(corners, cam)
        if rectangle is not None:
            view[label.name] = _Seen(rectangle, corners)
    return view


@cache
def _background(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), _SKY_BGR, np.uint8)
    image[height // 2:] = _ROAD_BGR
    return image


def _render(view: dict[str, _Seen], cam: CameraModel) -> bytes:
    """Flat-shaded boxes, farthest first, through the same projection as the oracle."""
    image = _background(cam.width, cam.height).copy()
    for name, seen in sorted(view.items(), key=lambda item: -item[1].corners[:, 0].max()):
        color = _BOX_BGR[_ACTORS[name].box_type]
        pieces = wireframe(seen.corners, cam, samples_per_edge=4)
        if pieces:
            hull = cv2.convexHull(np.round(np.vstack(pieces)).astype(np.int32))
            cv2.fillConvexPoly(image, hull, color)
        top = project_camera_points(seen.corners[_TOP_FACE], cam)
        if top.valid.all():
            cv2.fillConvexPoly(image, np.round(top.uv).astype(np.int32), [int(0.75 * c) for c in color])
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _flat(transform: np.ndarray) -> list[float]:
    return transform.reshape(-1).tolist()


def _lidar_row(key: tuple, label: _Label, t: float, rng: np.random.Generator) -> tuple:
    actor = label.actor
    speed, acceleration = _motion(actor.position, t)
    if rng.random() < _NULL_LABEL_RATE:  # label metadata is absent as a whole, not per field
        speed = acceleration = (None, None, None)
    first, last = actor.no_points_s
    points = 0 if first <= t < last else int(3000.0 / (1.0 + np.hypot(label.center[0], label.center[1])))
    return (*key, OBJECT_IDS[label.name], *label.center, *actor.size, label.heading,
            _BOX_TYPE_IDS[actor.box_type], points, int(0.8 * points), *speed, *acceleration)


def _projected_rows(key: tuple, camera_id: int, view: dict[str, _Seen]) -> list[tuple]:
    rows = []
    for name, seen in view.items():
        u_min, u_max, v_min, v_max = seen.rectangle
        rows.append((*key, camera_id, OBJECT_IDS[name], (u_min + u_max) / 2.0, (v_min + v_max) / 2.0,
                     u_max - u_min, v_max - v_min, _BOX_TYPE_IDS[_ACTORS[name].box_type]))
    return rows


def _image_row(key: tuple, camera: str, t: float, view: dict[str, _Seen], cam: CameraModel,
               rng: np.random.Generator) -> tuple:
    _, timestamp = key
    capture = t + IMAGE_OFFSET_S[camera]
    velocity = _motion(_ego_position, capture)[0]
    pose_time = timestamp / 1e6 + IMAGE_OFFSET_S[camera]
    shutter = rng.uniform(0.001, 0.010)
    trigger = pose_time - _POSE_LAG_S - (shutter + _READOUT_S) / 2.0
    return (*key, CAMERA_IDS[camera], _render(view, cam), _flat(_ego_pose(capture)), *velocity,
            pose_time, shutter, trigger, trigger + shutter + _READOUT_S)


def _synced_row(key: tuple, label: _Label, t: float, views: dict[str, dict[str, _Seen]]) -> tuple | None:
    """The center at the most visible camera's capture time, in that image's vehicle frame."""
    cameras = [camera for camera, view in views.items() if label.name in view]
    if not cameras:
        return None
    best = max(cameras, key=lambda camera: views[camera][label.name].area)
    capture = t + IMAGE_OFFSET_S[best]
    center = transform_points(inv_se3(_ego_pose(capture)), label.actor.position(capture))[0]
    return (*key, OBJECT_IDS[label.name], CAMERA_IDS[best], *center, *label.actor.size, label.heading)


def _arrow_type(name: str) -> pa.DataType:
    if "linear_velocity" in name:
        return pa.float32()
    return _ARROW_TYPES.get(name.split(".")[-1], pa.float64())


def _write_component(root: Path, ref: SegmentRef, tag: str, rows: list[tuple], rng: np.random.Generator) -> None:
    """Write rows, ordered as in columns.py. Shuffled, because real files don't share an order."""
    names = [*REQUIRED_COLUMNS[tag], *OPTIONAL_COLUMNS.get(tag, [])]
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    arrays = [pa.array(values, _arrow_type(name)) for name, values in zip(names, zip(*shuffled))]
    # `index` is not a Waymo column: it proves that extras are tolerated.
    table = pa.Table.from_arrays([*arrays, pa.array(np.arange(len(rows)))], names=[*names, "index"])
    path = component_path(root, ref, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_synthetic_segment(root, split="training", context_name="synthetic_cut_in", n_frames=199,
                            cameras=("FRONT", "FRONT_LEFT", "FRONT_RIGHT"), seed=0) -> SegmentRef:
    """Write the six Stage 1 components. `n_frames` counts frames written, after the two drops."""
    ref = SegmentRef(split, context_name)
    rng = np.random.default_rng(seed)
    models = {camera: _camera_model(camera) for camera in cameras}
    rows = {tag: [] for tag in STAGE1_COMPONENTS}
    for camera, cam in models.items():
        rows[CAMERA_CALIBRATION].append((context_name, CAMERA_IDS[camera], *cam.intrinsic, _flat(cam.extrinsic),
                                         cam.width, cam.height, cam.rolling_shutter_direction))

    timestamps = _frame_timestamps(n_frames, rng).tolist()
    for timestamp in timestamps:
        t = (timestamp - timestamps[0]) / 1e6
        label_time = t + LABEL_OFFSET_S
        key = (context_name, timestamp)
        world_from_vehicle = _ego_pose(label_time)
        labels = _labels(label_time, world_from_vehicle)
        rows[VEHICLE_POSE].append((*key, _flat(world_from_vehicle)))
        rows[LIDAR_BOX] += [_lidar_row(key, label, label_time, rng) for label in labels]

        views = {}
        for camera, cam in models.items():
            image_pose = _ego_pose(t + IMAGE_OFFSET_S[camera])
            views[camera] = _view(labels, camera_from_label(world_from_vehicle, image_pose, cam), cam)
            rows[PROJECTED_BOX] += _projected_rows(key, CAMERA_IDS[camera], views[camera])
            rows[CAMERA_IMAGE].append(_image_row(key, camera, t, views[camera], cam, rng))
        synced = [_synced_row(key, label, t, views) for label in labels]
        rows[SYNCED_BOX] += [row for row in synced if row is not None]

    for tag, component_rows in rows.items():
        _write_component(Path(root), ref, tag, component_rows, rng)
    return ref

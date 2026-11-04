"""Ego and object motion at full rate: central differences, the ego path, and two tables.

Labels sit in a vehicle frame that yaws with the ego, so differencing raw centers invents
lateral motion: d(y)/dt picks up -yaw_rate * x, a false 0.9 m/s at 54 m in one public sample.
So differences are taken in world axes, per object, over real timestamps, and only then
rotated into ego axes.

Never use the synced_* columns for trajectories: each is shifted to a per-object capture time.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .columns import BOX_TYPES, OBJECT_ID, TIME
from .config import MotionConfig
from .geometry import wrap_angle, yaw_of
from .loader import Poses

_JITTER_TOLERANCE = 0.95  # 10 Hz spacing varies, so a 0.2 s baseline can span 0.1996 s
_MIN_PATH_STEP_M = 0.05
_MIN_TRAVEL_M = 1.0       # poses drift while parked, and that's no direction of travel
_OTHER_LEVEL_M = 4.0      # a box center this far from the path's height is on an overpass, say, not in the lane
_LOCATE_CHUNK = 4096      # rows * path points would otherwise reach about 2 GB on a busy segment

_FOOTPRINT_SIGNS = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
_SYNCED_NAMES = {
    "most_visible_camera": "synced_camera",
    **{f"{part}_{axis}": f"synced_{part}_{axis}_m" for part in ("center", "size") for axis in "xyz"},
    "heading": "synced_heading_rad",
}


def central_difference(values, t_s, baseline_s, max_baseline_s) -> tuple[np.ndarray, np.ndarray]:
    """(d(values)/dt, the time each result describes), for `values` of shape (N,) or (N, k).

    Single steps are too noisy: in one public sample they read 7.37 and 8.51 m/s against a
    label of 7.97. So the window widens until it spans `baseline_s`, clipped to the array,
    which makes the ends one-sided. If the first span that qualifies is longer than
    `max_baseline_s`, or none does, the result is NaN: such a difference means too little.

    A result describes its pair's midpoint, which at the ends and beside a dropped frame
    isn't the sample's own time. A second pass must run over those times: over t_s, constant
    braking reads 20% low at the ends.
    """
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    result, midpoints = np.full(values.shape, np.nan), t.copy()
    for i in range(len(t)):
        for w in range(1, len(t)):
            a, b = max(i - w, 0), min(i + w, len(t) - 1)
            span = t[b] - t[a]
            if span >= _JITTER_TOLERANCE * baseline_s:
                if span <= max_baseline_s:
                    result[i] = (values[b] - values[a]) / span
                    midpoints[i] = (t[a] + t[b]) / 2.0
                break
    return result, midpoints


def _thin(points: np.ndarray) -> np.ndarray:
    """Drop points too close to the last one kept, so an ego creeping in tiny steps keeps its path."""
    keep, last = [0], points[0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - last) >= _MIN_PATH_STEP_M:
            keep.append(i)
            last = points[i]
    return points[keep]


class EgoPath:
    """The ego's own driven path in segment_local: post hoc, and a proxy for lane position.

    Past its ends it runs straight along the first and last headings, never along the first and
    last steps. A step can be 5 cm of pose noise, and 200 m of extension would swing on it. A
    heading is the way the nose points, so a reversing ego's extensions run against its travel.
    That's rare, and samples out there say they're off the driven path.
    """

    def __init__(self, points_xy: np.ndarray, start_yaw_rad: float, end_yaw_rad: float, extend_m: float = 200.0):
        points = _thin(np.asarray(points_xy, dtype=np.float64))
        if np.linalg.norm(points - points[0], axis=1).max() < _MIN_TRAVEL_M:
            points = points[:1]
        before, after = (np.array([np.cos(yaw), np.sin(yaw)]) for yaw in (start_yaw_rad, end_yaw_rad))
        line = np.vstack([points[0] - extend_m * before, points, points[-1] + extend_m * after])

        self.length_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        pieces = np.diff(line, axis=0)
        self._starts = line[:-1]
        self._lengths = np.linalg.norm(pieces, axis=1)
        self._directions = pieces / self._lengths[:, None]
        self._s_at_starts = np.cumsum(self._lengths) - self._lengths - extend_m  # s = 0 at the first pose

    def locate(self, points_xy) -> tuple[np.ndarray, np.ndarray]:
        """(s, d) at each point's nearest place on the path. d is signed, left of travel positive."""
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        located = [self._nearest(chunk) for chunk in np.array_split(points, len(points) // _LOCATE_CHUNK + 1)]
        return tuple(map(np.concatenate, zip(*located)))

    def _nearest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        relative = points[:, None] - self._starts                        # (points, pieces, 2)
        along = np.clip((relative * self._directions).sum(axis=2), 0.0, self._lengths)
        offsets = relative - along[..., None] * self._directions
        nearest = (offsets * offsets).sum(axis=2).argmin(axis=1)

        rows = np.arange(len(points))
        x, y = offsets[rows, nearest].T
        ux, uy = self._directions[nearest].T
        # Not the cross product alone: outside a bend the offset isn't square to the piece.
        return self._s_at_starts[nearest] + along[rows, nearest], np.sign(ux * y - uy * x) * np.hypot(x, y)


def pose_rows(poses: Poses, timestamps) -> np.ndarray:
    """Row of each timestamp in `poses`, by exact int64 match."""
    rows = pd.Index(poses.timestamps).get_indexer(timestamps)
    if (rows < 0).any():  # -1 would otherwise read the last pose
        raise ValueError("a timestamp has no vehicle_pose")
    return rows


def _axis_columns(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """One named column per axis: _axis_columns("v_{}_mps", xy) gives v_x_mps and v_y_mps."""
    return {name.format(axis): values[:, i] for i, axis in enumerate("xyz"[:values.shape[1]])}


def _to_ego_axes(rotation: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """First two entries of R.T @ (x, y, 0), row by row. R is vehicle -> world, so R.T undoes it."""
    return (rotation[:, :2, :2].transpose(0, 2, 1) @ world_xy[..., None])[..., 0]


def build_ego_table(poses: Poses, motion: MotionConfig) -> tuple[pd.DataFrame, EgoPath]:
    """One row per frame. Motion comes from poses rather than camera_image, so the label-only scan can share it."""
    t_s = (poses.timestamps - poses.timestamps[0]) / 1e6
    position = poses.world_from_vehicle[:, :3, 3] - poses.world_from_vehicle[0, :3, 3]
    yaw = np.array([yaw_of(pose) for pose in poses.world_from_vehicle])
    velocity, times = central_difference(position[:, :2], t_s, motion.velocity_baseline_s, motion.max_baseline_s)
    acceleration, _ = central_difference(velocity, times, motion.accel_baseline_s, motion.max_baseline_s)
    a_ego = _to_ego_axes(poses.world_from_vehicle[:, :3, :3], acceleration)
    path = EgoPath(position[:, :2], yaw[0], yaw[-1])

    table = pd.DataFrame({
        TIME: poses.timestamps,
        "t_s": t_s,
        "world_from_vehicle": poses.world_from_vehicle.reshape(-1, 16).tolist(),
        **_axis_columns("pos_local_{}_m", position),
        "yaw_rad": yaw,
        **_axis_columns("v_world_{}_mps", velocity),
        "speed_mps": np.hypot(velocity[:, 0], velocity[:, 1]),
        "a_long_mps2": a_ego[:, 0],
        "a_lat_mps2": a_ego[:, 1],
        "yaw_rate_radps": central_difference(np.unwrap(yaw), t_s, motion.velocity_baseline_s, motion.max_baseline_s)[0],
        "path_s_m": path.locate(position[:, :2])[0],
    })
    return table, path


def assign_aliases(boxes: pd.DataFrame) -> dict[str, str]:
    """Readable names such as V012, by first appearance then ID. Stage 2 validates against them."""
    first_rows = boxes.sort_values([TIME, OBJECT_ID]).drop_duplicates(OBJECT_ID)
    aliases, counts = {}, Counter()
    for object_id, code in zip(first_rows[OBJECT_ID], first_rows["type"]):
        prefix = BOX_TYPES[code][0].upper()
        counts[prefix] += 1
        aliases[object_id] = f"{prefix}{counts[prefix]:03d}"
    return aliases


def _footprints(center_xy: np.ndarray, size_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """(N, 4, 2) footprint corners of each box, in the frame of `center_xy` and `yaw`."""
    x, y = np.moveaxis(_FOOTPRINT_SIGNS * size_xy[:, None] / 2.0, -1, 0)
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return center_xy[:, None] + np.stack([cos * x - sin * y, sin * x + cos * y], axis=-1)


def _path_clearance(path: EgoPath, corners_local: np.ndarray) -> np.ndarray:
    """Distance from the path to the nearest corner, or 0 if the corners lie on both sides of it."""
    offsets = path.locate(corners_local.reshape(-1, 2))[1].reshape(-1, 4)
    straddling = (offsets.min(axis=1) < 0.0) & (offsets.max(axis=1) > 0.0)
    return np.where(straddling, 0.0, np.abs(offsets).min(axis=1))


def _per_object(values, t_s, objects, baseline_s, max_baseline_s) -> tuple[np.ndarray, np.ndarray]:
    """central_difference over each object's own rows, which interleave with everyone else's."""
    result, midpoints = np.full(values.shape, np.nan), t_s.copy()
    for rows in objects:
        result[rows], midpoints[rows] = central_difference(values[rows], t_s[rows], baseline_s, max_baseline_s)
    return result, midpoints


def _path_height(ego: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    """The ego's height where it passed each s. np.interp needs s sorted, and a stop's pose noise can step it back."""
    driven = ego.sort_values("path_s_m")
    return np.interp(s, driven["path_s_m"], driven["pos_local_z_m"])


def _camera_columns(boxes: pd.DataFrame, synced: pd.DataFrame, projected: pd.DataFrame) -> dict:
    """fov_cameras and synced_*. Left joins, so boxes with no camera row stay and their keys stay int64."""
    keys = [TIME, OBJECT_ID]
    joined = boxes[keys].merge(synced.rename(columns=_SYNCED_NAMES), how="left", on=keys, validate="one_to_one")
    in_view = projected.groupby(keys)["camera"].agg(list).to_dict()
    return {
        "fov_cameras": [in_view.get(key, []) for key in zip(boxes[TIME], boxes[OBJECT_ID])],
        **{name: joined[name].to_numpy() for name in _SYNCED_NAMES.values()},
        # The join widened this to float64 to hold its NaN.
        "synced_camera": joined["synced_camera"].fillna(-1).to_numpy(dtype=np.int8),
    }


def build_tracks_table(boxes, synced, projected, poses, ego, path, motion, rear_axle_to_front_m) -> pd.DataFrame:
    """One row per lidar_box row. Nothing is filtered here: that's for views."""
    ts = boxes[TIME].to_numpy()
    t_s = (ts - poses.timestamps[0]) / 1e6
    rows = pose_rows(poses, ts)
    at_frame = ego.iloc[rows]
    rotation, translation = poses.world_from_vehicle[rows, :3, :3], poses.world_from_vehicle[rows, :3, 3]

    center = boxes[["center_x", "center_y", "center_z"]].to_numpy()
    size = boxes[["size_x", "size_y", "size_z"]].to_numpy()
    heading = boxes["heading"].to_numpy()
    pos_world = (rotation @ center[..., None])[..., 0] + translation
    pos_local = pos_world - poses.world_from_vehicle[0, :3, 3]
    yaw_world = wrap_angle(heading + at_frame["yaw_rad"].to_numpy())

    objects = list(boxes.groupby(OBJECT_ID).indices.values())
    v_label = boxes[["speed_x", "speed_y"]].to_numpy()
    v_diff, diff_times = _per_object(pos_world[:, :2], t_s, objects, motion.velocity_baseline_s, motion.max_baseline_s)
    has_label, has_diff = np.isfinite(v_label).all(axis=1), np.isfinite(v_diff).all(axis=1)
    v_world = np.where(has_label[:, None], v_label, v_diff)
    v_times = np.where(has_label, t_s, diff_times)  # a label describes its own frame
    a_world, _ = _per_object(v_world, v_times, objects, motion.accel_baseline_s, motion.max_baseline_s)
    v_ego = at_frame[["v_world_x_mps", "v_world_y_mps"]].to_numpy()
    speed = np.hypot(v_world[:, 0], v_world[:, 1])

    s, lateral = path.locate(pos_local[:, :2])
    on_path = (s >= 0.0) & (s <= path.length_m)
    # Only where the ego drove: past the ends its height is unknown, and a hill would pass for a level.
    other_level = on_path & (np.abs(pos_local[:, 2] - _path_height(ego, s)) > _OTHER_LEVEL_M)
    clearance = _path_clearance(path, _footprints(pos_local[:, :2], size[:, :2], yaw_world))
    gap = _footprints(center[:, :2], size[:, :2], heading)[..., 0].min(axis=1) - rear_axle_to_front_m
    num_top_points = boxes["num_top_lidar_points"].to_numpy()

    return pd.DataFrame({
        TIME: ts,
        "t_s": t_s,
        OBJECT_ID: boxes[OBJECT_ID].to_numpy(),
        "alias": boxes[OBJECT_ID].map(assign_aliases(boxes)).to_numpy(),
        "type": boxes["type"].map(BOX_TYPES).to_numpy(),
        **_axis_columns("center_{}_m", center),
        **_axis_columns("size_{}_m", size),
        "heading_rad": heading,
        **_axis_columns("pos_world_{}_m", pos_world),
        **_axis_columns("pos_local_{}_m", pos_local),
        "yaw_world_rad": yaw_world,
        **_axis_columns("v_label_world_{}_mps", v_label),
        **_axis_columns("v_diff_world_{}_mps", v_diff),
        **_axis_columns("v_world_{}_mps", v_world),
        "v_source": np.select([has_label, has_diff], ["label", "central_difference"], "none"),
        **_axis_columns("a_world_{}_mps2", a_world),
        **_axis_columns("v_abs_ego_{}_mps", _to_ego_axes(rotation, v_world)),
        **_axis_columns("v_rel_ego_{}_mps", _to_ego_axes(rotation, v_world - v_ego)),
        **_axis_columns("a_abs_ego_{}_mps2", _to_ego_axes(rotation, a_world)),
        "speed_mps": speed,
        "stationary": speed < motion.stationary_mps,  # False for NaN, as an unknown speed should be
        "along_path_m": s - at_frame["path_s_m"].to_numpy(),
        "lateral_offset_m": lateral,
        "on_driven_path": on_path,
        "path_clearance_m": np.where(other_level, np.nan, clearance),
        "front_gap_m": np.where(gap > 0.0, gap, np.nan),
        "num_lidar_points": boxes["num_lidar_points"].to_numpy(),
        "num_top_lidar_points": num_top_points,
        "not_directly_observed": num_top_points == 0,
        **_camera_columns(boxes, synced, projected),
    })

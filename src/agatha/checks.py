"""Stage 1 acceptance checks: the numbered list in section 6 of the setup doc, as code that can fail.

Everything returning a dict with a `status` holds aggregates for `Report.add`, keyed by lowercase
names, because that's all the report takes. box_residuals, pick_review_frames, and output_hashes
hold row-level detail, so what they return stays on PACE.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import (
    CAMERA_IDS, CAMERA_IMAGE, CAMERA_NAMES, LIDAR_BOX, OBJECT_ID, PROJECTED_BOX, ROLLING_SHUTTER, SYNCED_BOX, TIME,
    VEHICLE_POSE,
)
from .geometry import box_rectangle, camera_from_label, row_corners, transform_from_list, transform_points
from .render import DRAWN_TYPES, uses_synced
from .sampling import gap_report

_FRAME = [TIME, "camera"]
_KEYS = {
    LIDAR_BOX: [TIME, OBJECT_ID],
    CAMERA_IMAGE: _FRAME,
    SYNCED_BOX: [TIME, OBJECT_ID],
    PROJECTED_BOX: [TIME, "camera", OBJECT_ID],
}

_EDGES = ["e_u_min", "e_u_max", "e_v_min", "e_v_max"]
_RESIDUAL_COLUMNS = [TIME, "camera", "ego_speed_mps", "ego_yaw_rate_radps", "object_speed_mps", "column_px",
                     "image_width_px", *_EDGES, "kind"]
_LABEL_COLUMNS = [
    *[f"{part}_{axis}_m" for part in ("center", "size") for axis in "xyz"], "heading_rad",
    *[f"synced_{part}_{axis}_m" for part in ("center", "size") for axis in "xyz"], "synced_heading_rad",
    "synced_camera", "speed_mps", "type",
]
_SPEED_BINS = ("slow", "under_5", "under_15", "from_15")  # by upper edge in m/s: slow_speed_mps sets the first
_COLUMN_BINS = ("column_1", "column_2", "column_3", "column_4", "column_5")

_MOVING_MPS = 1.0  # a stopped object's label is near zero in any axes, so it can't separate the readings
_EGO_VELOCITY_LIMIT_MPS = 1.0  # a wrong axis, unit, or clock is off by about the ego's speed. Unmeasured, like the rest
_SLOW_YAW_RATE_RADPS = 0.05  # yaw moves the image during readout at any range: about 2 px at the edges here, 7 at 0.15
_SYNCED_MARGIN_PX = 5.0  # how far a synced box may trail the label on stationary objects. Unmeasured, like the rest
_MIN_SPREAD_BOXES = 5  # with fewer, a spread's four numbers come close to being the rows, and those stay on PACE
_REVIEW_RANDOM, _REVIEW_PER_REASON = 12, 6


def _spread(values) -> dict:
    """Min, median, p95, and max, skipping NaN. Empty gives NaN, which the report keeps as None."""
    series = pd.Series(values, dtype=np.float64)
    return {"min": series.min(), "median": series.median(), "p95": series.quantile(0.95), "max": series.max()}


def _camera_key(camera: int) -> str:
    return CAMERA_NAMES[camera].lower()


def _rows_without_box(table: pd.DataFrame, boxes: pd.DataFrame) -> int:
    keys = [TIME, OBJECT_ID]
    return int((~pd.MultiIndex.from_frame(table[keys]).isin(pd.MultiIndex.from_frame(boxes[keys]))).sum())


def check_joins(poses, boxes, synced, projected, image_meta, camera_ids) -> dict:
    """Check 1: every key is int64, matches exactly, and appears once. Nearly equal is an orphan."""
    tables = dict(zip(_KEYS, (boxes, image_meta, synced, projected)))
    rows_without_pose = {tag: int((~table[TIME].isin(poses.timestamps)).sum()) for tag, table in tables.items()}
    rows_without_box = {tag: _rows_without_box(tables[tag], boxes) for tag in (SYNCED_BOX, PROJECTED_BOX)}
    duplicate_keys = {tag: int(table.duplicated(_KEYS[tag]).sum()) for tag, table in tables.items()}
    duplicate_keys[VEHICLE_POSE] = len(poses.timestamps) - len(np.unique(poses.timestamps))
    timestamps = [poses.timestamps, *[table[TIME] for table in tables.values()]]
    non_int64_keys = sum(column.dtype != np.int64 for column in timestamps)
    # No orphans and no duplicates is also what a camera that loaded nothing looks like.
    cameras_without_images = len(set(camera_ids) - set(image_meta["camera"]))

    counts = [*rows_without_pose.values(), *rows_without_box.values(), *duplicate_keys.values(),
              non_int64_keys, cameras_without_images]
    return {
        "status": "fail" if any(counts) else "no_data" if boxes.empty else "pass",
        "rows_without_pose": rows_without_pose,
        "rows_without_box": rows_without_box,
        "duplicate_keys": duplicate_keys,
        "non_int64_keys": non_int64_keys,
        "cameras_without_images": cameras_without_images,
    }


def check_gaps(poses, image_meta, config) -> dict:
    """Check 2: spacing, and how many pose frames each camera lacks.

    A frame gone from the whole segment is normal, so gaps are only counted. A frame with a
    pose but no image from one camera is a hole in that camera, and that fails.
    """
    gaps = gap_report(poses.timestamps, config.checks.gap_ms)
    frames_missing = {_camera_key(camera): int(np.isin(poses.timestamps, times, invert=True).sum())
                      for camera, times in image_meta.groupby("camera")[TIME]}
    status = "no_data" if gaps.n_frames < 2 or image_meta.empty else "fail" if any(frames_missing.values()) else "pass"
    return {"status": status, **gaps._asdict(), "frames_missing": frames_missing}


def _box_rectangles(box, cam) -> dict:
    """Rectangles for one box, by kind.

    lidar_box goes through the image pose. frame_pose is the reading to hold it against: at
    speed, a side camera's stationary objects are where a pose bug shows. camera_synced_box
    exists if this is the box's camera and its values are there. drawn is what
    render.draw_boxes shows, by its same tests: the synced box or else the label, and nothing
    for a type it skips. A None from box_rectangle can't choose between them: render still
    draws a box with a bad corner. label_fallback is the drawn boxes with no synced box to
    stand on.
    """
    corners = row_corners(box)
    camera_from_frame = camera_from_label(transform_from_list(box.world_from_vehicle),
                                          transform_from_list(box.world_from_image_vehicle), cam)
    label = box_rectangle(transform_points(camera_from_frame, corners), cam)
    rectangles = {"lidar_box": label,
                  "frame_pose": box_rectangle(transform_points(cam.camera_from_vehicle, corners), cam)}
    has_synced = uses_synced(box, box.camera)
    if has_synced:
        # Extrinsic only, Waymo's own path: going through the poses would count ego motion twice.
        rectangles["camera_synced_box"] = box_rectangle(
            transform_points(cam.camera_from_vehicle, row_corners(box, "synced_")), cam)
    if box.type in DRAWN_TYPES:
        if not has_synced:
            rectangles["label_fallback"] = label
        rectangles["drawn"] = rectangles["camera_synced_box"] if has_synced else label
    return rectangles


def _inside(rectangle, cam, margin: float) -> bool:
    u_min, u_max, v_min, v_max = rectangle
    return u_min >= margin and u_max <= cam.width - margin and v_min >= margin and v_max <= cam.height - margin


def box_residuals(segment, projected, cameras, config) -> pd.DataFrame:
    """Signed per-edge errors against projected_lidar_box, the numeric oracle. Never leaves PACE.

    Waymo clamps its rectangles to the image, so an edge at the border says where the image
    ends, not where the box projects. Only boxes well inside on both sides qualify. Joins are
    inner because check_joins counts whatever has no match. Each row carries `image_width_px`
    because check_residuals bins by column and has no other way to know the width.
    """
    margin = config.checks.border_margin_px
    ego = segment.ego[[TIME, "speed_mps", "yaw_rate_radps"]].rename(
        columns={"speed_mps": "ego_speed_mps", "yaw_rate_radps": "ego_yaw_rate_radps"})
    joined = (projected
              .merge(segment.frames[[*_FRAME, "world_from_vehicle", "world_from_image_vehicle"]], on=_FRAME)
              .merge(ego, on=TIME)
              .merge(segment.tracks[[TIME, OBJECT_ID, *_LABEL_COLUMNS]], on=[TIME, OBJECT_ID],
                     suffixes=("_projected", "")))  # both tables have a type, and render reads the tracks'
    rows = []
    for box in joined.itertuples():
        cam = cameras[box.camera]
        half_u, half_v = box.size_x / 2.0, box.size_y / 2.0
        oracle = (box.center_x - half_u, box.center_x + half_u, box.center_y - half_v, box.center_y + half_v)
        for kind, rectangle in _box_rectangles(box, cam).items():
            if rectangle is not None and _inside(rectangle, cam, margin) and _inside(oracle, cam, margin):
                rows.append((box.frame_timestamp_micros, box.camera, box.ego_speed_mps, box.ego_yaw_rate_radps,
                             box.speed_mps, box.center_x, cam.width, *np.subtract(rectangle, oracle), kind))
    return pd.DataFrame(rows, columns=_RESIDUAL_COLUMNS)


def _edge_spread(rows: pd.DataFrame) -> dict:
    """Sizes of the errors, pooled over the four edges, so each box counts four times, then their signed medians.

    A wrong pose or clock pushes boxes one way, which sizes can't show, and by column the
    signed u median crosses zero at the column where global shutter is exact.
    """
    u, v = (pd.Series(rows[edges].to_numpy().ravel(), dtype=np.float64) for edges in (_EDGES[:2], _EDGES[2:]))
    return {"boxes": len(rows), **_spread(pd.concat([u, v]).abs()), "signed_u_median": u.median(),
            "signed_v_median": v.median()}


def _by_camera(rows: pd.DataFrame) -> dict:
    return {_camera_key(camera): _edge_spread(group) for camera, group in rows.groupby("camera")}


def _by_bin(rows: pd.DataFrame, values: pd.Series, names, edges) -> dict:
    """_edge_spread of each [low, high) bin. A NaN value lands in none of them."""
    return {name: _edge_spread(rows[values.between(low, high, inclusive="left")])
            for name, low, high in zip(names, edges, edges[1:])}


def _over(spread: dict, median_limit: float, p95_limit: float) -> bool:
    """False for an empty bin, whose NaN compares false."""
    return bool(spread["median"] > median_limit or spread["p95"] > p95_limit)


def _stationary_by_camera(residuals: pd.DataFrame, stationary_mps: float) -> dict:
    """Readings on still objects, so object motion can't explain a synced box trailing the label.

    The label itself still carries rolling shutter. frame_pose is here because this is where
    the image pose should beat it: traffic moving with the ego can go the other way.
    """
    still = residuals[residuals["object_speed_mps"] < stationary_mps]
    return {_camera_key(camera): {kind: _edge_spread(rows[rows["kind"] == kind])
                                  for kind in ("lidar_box", "camera_synced_box", "frame_pose")}
            for camera, rows in still.groupby("camera")}


def _synced_trails(lidar_box: dict, camera_synced_box: dict) -> bool:
    """A synced box's frame is an inference no vector backs. If it were wrong, this is where it would show."""
    return bool(min(lidar_box["boxes"], camera_synced_box["boxes"]) >= _MIN_SPREAD_BOXES
                and camera_synced_box["p95"] > lidar_box["p95"] + _SYNCED_MARGIN_PX)


def _reported(spreads: dict) -> dict:
    """What the report may hold of a spread, or of a dict of them. Verdicts come first, on the full numbers."""
    if "boxes" not in spreads:
        return {name: _reported(spread) for name, spread in spreads.items()}
    return spreads if spreads["boxes"] >= _MIN_SPREAD_BOXES else {"boxes": spreads["boxes"]}


def check_residuals(residuals, config) -> dict:
    """Check 3. The limits judge the boxes an image shows wholly inside its border.

    Waymo clamps the oracle at the border, so nothing past it can be measured, and that leaves
    out most boxes the label draws: a box outside its most visible camera usually straddles it.

    Not lidar_box alone: it ignores rolling shutter, so at speed it can't be exact. On the side
    cameras at 29 m/s it measures 68 to 80 px at p95, where the drawn boxes reach 3 to 5 px.
    Synced boxes on all cameras reach 0.62 px, most of them moving at 15 m/s or more, which
    shows the oracle carries object motion. lidar_box stays in the report and the bins, because
    it's the path that shows pose and timing.

    FRONT's lidar_box must clear the overall limits too. A synced box never reads center_*, so
    at speed nothing else can be counted on to judge the table Stage 4 reads. FRONT can carry
    that because its image pose is the frame pose: rolling shutter is all that's left, p95 3.3
    px at 29 m/s.

    Each camera must clear the overall limits alone, as check 7 asks. Pooled, a camera with
    few boxes could be any amount wrong and still pass. A configured camera with no boxes
    has shown nothing either way, so the most it can give is no_data.

    The slow limits judge lidar_box, because on a slow ego it should be exact on anything
    standing still. They need a steady ego as well as a slow one, because yaw alone moves the
    image during readout. They leave out moving objects, because the oracle moves those and the
    label doesn't, so a stopped ego beside cross traffic would otherwise fail on correct code.
    They can't be relied on for a pose bug: that slow, the two poses are millimeters apart.
    """
    limits = config.checks
    lidar = residuals[residuals["kind"] == "lidar_box"]
    drawn = residuals[residuals["kind"] == "drawn"]
    overall = _edge_spread(drawn)
    by_camera = _by_camera(drawn)
    front_label = _edge_spread(lidar[lidar["camera"] == CAMERA_IDS["FRONT"]])
    cameras_without_boxes = len(set(config.cameras) - set(drawn["camera"]))
    speed_edges = [0.0, limits.slow_speed_mps, 5.0, 15.0, np.inf]
    column_bin = len(_COLUMN_BINS) * lidar["column_px"] / lidar["image_width_px"]
    slow_and_steady = _edge_spread(lidar[(lidar["ego_speed_mps"] < limits.slow_speed_mps)
                                         & (lidar["ego_yaw_rate_radps"].abs() < _SLOW_YAW_RATE_RADPS)
                                         & (lidar["object_speed_mps"] < config.motion.stationary_mps)])
    stationary = _stationary_by_camera(residuals, config.motion.stationary_mps)
    failed = (any(_over(spread, limits.residual_median_px, limits.residual_p95_px)
                  for spread in (overall, *by_camera.values(), front_label))
              or _over(slow_and_steady, limits.slow_residual_median_px, limits.slow_residual_p95_px)
              or any(_synced_trails(spreads["lidar_box"], spreads["camera_synced_box"])
                     for spreads in stationary.values()))
    return {
        "status": "fail" if failed else "no_data" if drawn.empty or cameras_without_boxes else "pass",
        "cameras_without_boxes": cameras_without_boxes,
        **_reported({
            "drawn": overall,
            "by_camera": by_camera,
            "label_fallback_by_camera": _by_camera(residuals[residuals["kind"] == "label_fallback"]),
            "lidar_box": _edge_spread(lidar),
            "lidar_box_by_camera": _by_camera(lidar),
            "camera_synced_box": _edge_spread(residuals[residuals["kind"] == "camera_synced_box"]),
            "frame_pose_by_camera": _by_camera(residuals[residuals["kind"] == "frame_pose"]),
            "stationary_by_camera": stationary,
            "slow_and_steady": slow_and_steady,
            "by_ego_speed": _by_bin(lidar, lidar["ego_speed_mps"], _SPEED_BINS, speed_edges),
            "by_object_speed": _by_bin(lidar, lidar["object_speed_mps"], _SPEED_BINS, speed_edges),
            "by_column": _by_bin(lidar, column_bin, _COLUMN_BINS, range(len(_COLUMN_BINS) + 1)),
        }),
    }


def check_tracks(segment, config) -> dict:
    """Check 4. Real tracks have gaps and type changes, so those are counted. Only a jump no object could make fails."""
    tracks = segment.tracks
    objects = tracks[OBJECT_ID]
    by_object = tracks.groupby(OBJECT_ID)

    # Counted in frames that exist, not in time: a frame dropped from the segment is no gap in a track.
    frame_index = pd.Series(pd.Index(segment.ego[TIME]).get_indexer(tracks[TIME]), index=tracks.index)
    frames_skipped = frame_index.groupby(objects).diff() - 1
    gaps = frames_skipped[frames_skipped > 0]

    previous_type = by_object["type"].shift()
    type_changed = previous_type.notna() & (previous_type != tracks["type"])

    step_m = np.hypot(by_object["pos_world_x_m"].diff(), by_object["pos_world_y_m"].diff())
    implied_mps = step_m / (by_object[TIME].diff() / 1e6)
    spikes = implied_mps[implied_mps > config.checks.speed_spike_mps]

    return {
        "status": "no_data" if tracks.empty else "fail" if len(spikes) else "pass",
        "tracks": by_object.ngroups,
        "rows": len(tracks),
        "length_frames": _spread(by_object.size()),
        "internal_gaps": {"count": len(gaps), "tracks": objects[gaps.index].nunique(), "frames": _spread(gaps)},
        "type_changes": {"count": int(type_changed.sum()), "tracks": objects[type_changed].nunique()},
        "speed_spikes": {"count": len(spikes), "tracks": objects[spikes.index].nunique()},
        "zero_point_rows": int((tracks["num_lidar_points"] == 0).sum()),
        "zero_top_point_rows": int(tracks["not_directly_observed"].sum()),
    }


def check_velocity(segment) -> dict:
    """Check 5: is Waymo's `speed` in world axes? Three readings of the label, against position differences."""
    tracks = segment.tracks.merge(segment.ego[[TIME, "yaw_rad", "v_world_x_mps", "v_world_y_mps"]], on=TIME,
                                  suffixes=("", "_ego"))
    label = tracks[["v_label_world_x_mps", "v_label_world_y_mps"]].to_numpy()
    difference = tracks[["v_diff_world_x_mps", "v_diff_world_y_mps"]].to_numpy()
    ego = tracks[["v_world_x_mps_ego", "v_world_y_mps_ego"]].to_numpy()
    cos, sin = np.cos(tracks["yaw_rad"].to_numpy()), np.sin(tracks["yaw_rad"].to_numpy())
    vehicle_axes = np.column_stack([cos * label[:, 0] - sin * label[:, 1], sin * label[:, 0] + cos * label[:, 1]])
    # Near due east, vehicle axes are world axes, so those rows can't tell the two apart.
    usable = (np.isfinite(ego).all(axis=1) & (np.hypot(difference[:, 0], difference[:, 1]) > _MOVING_MPS)
              & (np.linalg.norm(vehicle_axes - label, axis=1) > _MOVING_MPS))  # NaN compares false
    if not usable.any():
        return {"status": "no_data", "rows": 0}

    world, vehicle, ego_relative = [float(np.median(np.linalg.norm((difference - reading)[usable], axis=1)))
                                    for reading in (label, vehicle_axes, vehicle_axes + ego)]
    return {
        "status": "pass" if world < 0.5 * min(vehicle, ego_relative) else "fail",
        "rows": int(usable.sum()),
        "median_residual_world_mps": world,
        "median_residual_vehicle_mps": vehicle,
        "median_residual_ego_relative_mps": ego_relative,
    }


def check_ego_velocity(segment, image_meta) -> dict:
    """The ego table's pose differences against FRONT's linear_velocity, which is in world axes.

    It's the one reading of ego motion here that doesn't come from differencing these poses.
    It isn't the source because the label-only scan has no camera_image and must run the same code.
    """
    front = image_meta[image_meta["camera"] == CAMERA_IDS["FRONT"]].merge(
        segment.ego[[TIME, "v_world_x_mps", "v_world_y_mps"]], on=TIME)
    residuals = np.hypot(front["linear_velocity_x_mps"] - front["v_world_x_mps"],
                         front["linear_velocity_y_mps"] - front["v_world_y_mps"]).dropna()
    spread = _spread(residuals)
    return {
        "status": "no_data" if residuals.empty else "fail" if spread["median"] > _EGO_VELOCITY_LIMIT_MPS else "pass",
        "rows": len(residuals),
        "residual_mps": spread,
    }


def timing_summary(frames, cameras) -> dict:
    """Capture timing in ms, and readout direction, per camera. No limit applies, so `pass` only means there's data."""
    trigger, done, pose = frames["trigger_time_s"], frames["readout_done_time_s"], frames["pose_timestamp_s"]
    timing = {
        "capture_offset_ms": frames["capture_offset_ms"],
        "exposure_mid_after_pose_ms": ((trigger + done) / 2.0 - pose) * 1e3,
        "readout_ms": (done - trigger - frames["shutter_s"]) * 1e3,
        "shutter_ms": frames["shutter_s"] * 1e3,
    }
    per_camera = {_camera_key(camera): {"rolling_shutter_direction": ROLLING_SHUTTER[cam.rolling_shutter_direction],
                                        **{name: _spread(values[frames["camera"] == camera])
                                           for name, values in timing.items()}}
                  for camera, cam in cameras.items()}
    return {"status": "no_data" if frames.empty else "pass", "cameras": per_camera}


def null_summary(boxes) -> dict:
    """How often labels lack speed or accel, judged on x and y, the axes tracks.py uses.

    An exact zero in speed_z is probably a filled-in null, so its share is of the rows with a value.
    The spec sets no limit here, so `pass` only means there's data.
    """
    return {
        "status": "no_data" if boxes.empty else "pass",
        "rows": len(boxes),
        "speed_null_share": boxes[["speed_x", "speed_y"]].isna().any(axis=1).mean(),
        "accel_null_share": boxes[["accel_x", "accel_y"]].isna().any(axis=1).mean(),
        "speed_z_zero_share": (boxes["speed_z"].dropna() == 0.0).mean(),
    }


def _top_frames(rows: pd.DataFrame, scores: pd.Series) -> list:
    """The highest-scoring frames, by each frame's best row: one frame's boxes shouldn't fill the list."""
    best = scores.groupby([rows[TIME], rows["camera"]]).max()
    return best.sort_values(ascending=False).index[:_REVIEW_PER_REASON].tolist()


def pick_review_frames(residuals, segment, seed=0) -> list[tuple[int, int, str]]:
    """Check 8: frames for Andrew's eyes, as (timestamp, camera, reason). A frame keeps its first reason.

    Random comes last, so a frame picked for a reason says so. Near and fast ranks speed over
    range, the inverse of seconds to reach the ego, which is high only when an object is both.
    """
    drawn = residuals[residuals["kind"] == "drawn"]
    edge_px = np.minimum(drawn["column_px"], drawn["image_width_px"] - drawn["column_px"])
    # fov_cameras comes from projected_lidar_box, which can name an image that isn't there: review needs the image.
    in_view = (segment.tracks.rename(columns={"fov_cameras": "camera"}).explode("camera").dropna(subset="camera")
               .astype({"camera": np.int64}).merge(segment.frames[_FRAME], on=_FRAME))
    near_fast = in_view["speed_mps"] / np.hypot(in_view["center_x_m"], in_view["center_y_m"])
    sampled = segment.frames[_FRAME].sample(n=min(_REVIEW_RANDOM, len(segment.frames)), random_state=seed)

    picks = {
        "worst_residual": _top_frames(drawn, drawn[_EDGES].abs().max(axis=1)),
        "image_edge": _top_frames(drawn, -edge_px),
        "near_fast": _top_frames(in_view, near_fast),
        "random": list(zip(sampled[TIME], sampled["camera"])),
    }
    reasons = {}
    for reason, frames in picks.items():
        for frame in frames:
            reasons.setdefault(frame, reason)
    return [(int(timestamp), int(camera), reason) for (timestamp, camera), reason in reasons.items()]


def output_hashes(directory) -> dict[str, str]:
    """Check 6: sha256 by relative path, except qa/ and clips/. Compare two runs, and report only whether they match.

    Paths and hashes stay on PACE, so this dict is the one Report.add should refuse.
    """
    directory = Path(directory)
    hashes = {}
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if path.is_file() and relative.parts[0] not in ("qa", "clips"):
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes

"""SceneWindows: pure views of the full-rate tables, so any range at any rate can be asked for."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import CAMERA_NAMES, LIDAR_COMPONENTS, STAGE1_COMPONENTS, TIME
from .contracts import (
    EgoFootprint, EgoSample, FrameRef, ObjectSample, ObjectTrack, OmittedObject, SceneWindow, SourceRef,
)
from .files import write_atomic
from .sampling import compute_windows, select_frames
from .store import SegmentOutputs

_ROAD_USERS = ("vehicle", "pedestrian", "cyclist")
_SELECTION_RULE = (
    "Objects that are a vehicle, pedestrian, or cyclist at any image frame, ranked by minimum "
    "path_clearance_m over the image frames, then minimum range, then alias. The first {} are "
    "kept, plus any the request named, and omitted_objects lists the rest, which the images still "
    "label. Stationary objects and objects outside every camera's FOV are not filtered out."
)


def _optional(value: float) -> float | None:
    return None if np.isnan(value) else value


def _frame_ref(row, profile: str) -> FrameRef:
    stem = f"render/{row.camera_name}/{row.frame_timestamp_micros}"
    return FrameRef(
        frame_timestamp_micros=row.frame_timestamp_micros,
        t_s=row.t_s,
        camera=row.camera_name,
        capture_offset_ms=row.capture_offset_ms,
        clean_ref=f"{stem}__clean.jpg",
        annotated_ref=f"{stem}__{profile}.jpg",
    )


def _object_sample(row, context_only: bool) -> ObjectSample:
    return ObjectSample(
        frame_timestamp_micros=row.frame_timestamp_micros,
        t_s=row.t_s,
        type=row.type,
        pos_ego_x_m=row.center_x_m,
        pos_ego_y_m=row.center_y_m,
        front_gap_m=_optional(row.front_gap_m),
        along_path_m=row.along_path_m,
        lateral_offset_m=row.lateral_offset_m,
        on_driven_path=row.on_driven_path,
        path_clearance_m=_optional(row.path_clearance_m),
        v_abs_ego_x_mps=_optional(row.v_abs_ego_x_mps),
        v_abs_ego_y_mps=_optional(row.v_abs_ego_y_mps),
        v_rel_ego_x_mps=_optional(row.v_rel_ego_x_mps),
        v_rel_ego_y_mps=_optional(row.v_rel_ego_y_mps),
        a_abs_ego_x_mps2=_optional(row.a_abs_ego_x_mps2),
        a_abs_ego_y_mps2=_optional(row.a_abs_ego_y_mps2),
        v_source=row.v_source,
        fov_cameras=[CAMERA_NAMES[camera] for camera in row.fov_cameras],
        not_directly_observed=row.not_directly_observed,
        stationary=row.stationary,
        context_only=context_only,
    )


def _object_track(whole: pd.DataFrame, sampled: np.ndarray, image_times: set[int], first_frame: int,
                  last_frame: int) -> ObjectTrack:
    """`whole` is the object's full track, because the summary fields describe all of it."""
    first, last = whole.iloc[0], whole.iloc[-1]
    return ObjectTrack(
        alias=first["alias"],
        laser_object_id=first["laser_object_id"],
        type=first["type"],
        first_t_s=first["t_s"],
        last_t_s=last["t_s"],
        range_at_first_seen_m=np.hypot(first["center_x_m"], first["center_y_m"]),
        truncated_start=bool(first[TIME] == first_frame),
        truncated_end=bool(last[TIME] == last_frame),
        samples=[_object_sample(row, row.frame_timestamp_micros not in image_times)
                 for row in whole[whole[TIME].isin(sampled)].itertuples()],
    )


def _omitted_object(whole: pd.DataFrame) -> OmittedObject:
    first = whole.iloc[0]
    return OmittedObject(alias=first["alias"], laser_object_id=first["laser_object_id"], type=first["type"])


def _ranked_aliases(in_images: pd.DataFrame) -> list[str]:
    """Road users by minimum path clearance over the image frames, then range: all that cross the path tie at 0."""
    road_users = in_images.loc[in_images["type"].isin(_ROAD_USERS), "alias"]
    rows = in_images[in_images["alias"].isin(road_users)]
    ranks = (rows.assign(range_m=np.hypot(rows["center_x_m"], rows["center_y_m"]))
             .groupby("alias", as_index=False)[["path_clearance_m", "range_m"]].min())
    return ranks.sort_values(["path_clearance_m", "range_m", "alias"])["alias"].tolist()


def build_scene_window(segment: SegmentOutputs, config, start_s, end_s, fps, cameras, profile,
                       closed_end=False, max_objects=None, context_s=None, force_include=(),
                       keep_aliases=()) -> SceneWindow:
    """One window of `cameras` (ids), with track-only context either side.

    `keep_aliases` stay whatever their rank, on top of max_objects, so Pass 2's candidate is always in its input.
    Pure: it writes nothing, and it only names images, which render.ensure_rendered makes.
    """
    meta, frames, ego, tracks = segment.meta, segment.frames, segment.ego, segment.tracks
    max_objects = config.max_objects if max_objects is None else max_objects
    context_s = config.context_s if context_s is None else context_s

    absent = set(cameras) - set(frames["camera"])
    if absent:  # frames holds only the cameras Stage 1 ran on, so anything else needs a rerun with them
        raise ValueError(f"Stage 1 has no frames for cameras {sorted(absent)}")
    images = frames[frames["camera"].isin(cameras)]
    selection = select_frames(images[TIME], meta["t0_micros"], fps, start_s, end_s, closed_end, force_include)
    sampled = select_frames(ego[TIME], meta["t0_micros"], fps, max(start_s - context_s, 0.0),
                            min(end_s + context_s, meta["last_t_s"]), closed_end=True,
                            force_include=selection.timestamps).timestamps

    ranked = _ranked_aliases(tracks[tracks[TIME].isin(selection.timestamps)])
    kept = list(dict.fromkeys([*ranked[:max_objects], *keep_aliases]))
    image_times = set(selection.timestamps.tolist())
    first_frame, last_frame = ego[TIME].iloc[[0, -1]]
    return SceneWindow(
        segment_context_name=meta["context_name"],
        split=meta["split"],
        t0_micros=meta["t0_micros"],
        window_start_s=start_s,
        window_end_s=end_s,
        cameras=[CAMERA_NAMES[camera] for camera in sorted(cameras)],
        profile=profile,
        requested_fps=fps,
        achieved_fps=selection.achieved_fps,
        max_gap_ms=selection.max_gap_ms,
        frames=[_frame_ref(row, profile) for row in images[images[TIME].isin(selection.timestamps)].itertuples()],
        objects=[_object_track(tracks[tracks["alias"] == alias], sampled, image_times, first_frame, last_frame)
                 for alias in kept],
        omitted_objects=[_omitted_object(tracks[tracks["alias"] == alias]) for alias in ranked if alias not in kept],
        selection_rule=_SELECTION_RULE.format(max_objects),
        ego=[EgoSample(**{name: _optional(getattr(row, name)) for name in EgoSample.model_fields})
             for row in ego[ego[TIME].isin(sampled)].itertuples()],
        ego_footprint=EgoFootprint(rear_axle_to_front_m=meta["rear_axle_to_front_m"]),
        sources=[SourceRef(component=tag, split=meta["split"], context_name=meta["context_name"],
                           dataset_version=meta["dataset_version"]) for tag in (*STAGE1_COMPONENTS, *LIDAR_COMPONENTS)],
        preprocessing_version=meta["preprocessing_version"],
    )


def default_windows(segment: SegmentOutputs, config) -> list[SceneWindow]:
    """Pass 1's windows, from config. Pass 2 calls build_scene_window directly."""
    return [build_scene_window(segment, config, window.start_s, window.end_s, config.fps, config.cameras,
                               config.profile, closed_end=window.closed_end)
            for window in compute_windows(segment.meta["last_t_s"], config.window_s, config.stride_s)]


def write_scene_window(window: SceneWindow, directory) -> Path:
    """Write under the segment's directory, in a name that ends in a hash of the content.

    The rest of the name can't say everything that shaped a window: two Pass 2 candidates over
    the same range can cite different frames, and each must keep its own input for audit.
    """
    text = window.model_dump_json()
    name = (f"{round(window.window_start_s * 1e3)}_{round(window.window_end_s * 1e3)}"
            f"__{window.requested_fps:g}fps__{'+'.join(window.cameras)}__{window.profile}"
            f"__{hashlib.sha256(text.encode()).hexdigest()[:8]}.json")
    path = Path(directory) / "windows" / name
    write_atomic(path, text.encode() + b"\n")
    return path

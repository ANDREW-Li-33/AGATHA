"""The cut-in scan: rank labeled segments by how likely each holds a sudden cut-in, from labels alone.

It reads lidar_box, vehicle_pose, stats and camera_calibration, a few GB for all 1,000 labeled
segments, so images are fetched only for the best few. Perception v2 has no lane lines, so the
ego's own driven path stands in for its lane, measured by Stage 1's own table code. Labels can't
prove a cut-in: this finds candidates, and Andrew decides by eye on the review pages.

Real data passes through here. Library code prints nothing, and the CLI prints only counts and
paths that hold no names. Only scan.json leaves PACE, through report.Report. The tables,
cutin_top.txt and fast_left.txt name contexts, so they stay there.
"""

from __future__ import annotations

import base64
import html
import os
import re
import shutil
import signal
import sys
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .checks import _inside
from .columns import CAMERA_IDS, LIDAR_BOX, OBJECT_ID, STATS, TIME, TIMESTAMP, col
from .config import Stage1Config, load_config
from .files import write_atomic
from .geometry import CameraModel, box_rectangle, row_corners, transform_points, wrap_angle
from .loader import SegmentRef, component_path, load_cameras, load_lidar_boxes, load_poses, read_jpegs
from .render import boxes_for_image, render_annotated
from .report import Report, guarded
from .run_stage1 import _QuietParser
from .sampling import select_frames
from .store import SegmentOutputs, camera_models, load_segment
from .tracks import build_ego_table, build_tracks_table

# An event's columns, with types that hold even with no rows, so a quiet segment concatenates like any other.
_EVENT_TYPES = {
    OBJECT_ID: str, "alias": str,
    "onset_timestamp_micros": np.int64, "entry_timestamp_micros": np.int64,
    "onset_t_s": np.float64, "entry_t_s": np.float64,
    "headway_s": np.float64, "closing_mps": np.float64, "required_decel_mps2": np.float64,
    "ego_min_a_long_mps2": np.float64, "severity_mps2": np.float64,
    "crossing_s": np.float64, "peak_lateral_mps": np.float64, "own_move_m": np.float64,
    "lateral_change_mps2": np.float64, "past_path_end": bool,
}
_SEGMENT_KEYS = ["split", "context_name"]
_SCANNED_EVENT_TYPES = {**dict.fromkeys(_SEGMENT_KEYS, str), **_EVENT_TYPES}
_CONDITIONS = ("time_of_day", "weather", "location")
_SEGMENT_COLUMNS = [*_SEGMENT_KEYS, "n_frames", "ego_speed_median_mps", "ego_speed_max_mps", "vehicle_tracks", "events",
                    "best_headway_s", "left_at_speed", "front_left_calibrated", *_CONDITIONS]

# The own move's circle. Slower than this, yaw rate over speed is mostly noise, so the circle is a straight line.
_MIN_TURN_SPEED_MPS = 1.0
_MAX_CURVATURE = 0.2  # 1/m. No car turns tighter than about 5 m, so a noisy yaw rate can't bend the circle more.
# No car brakes much harder than 10 m/s², so a cap ten times that changes nothing a car could do, and a gap of a
# millimeter can't make a number the report refuses (1e12 and up), which would fail the whole scan. A floor under
# the gap would change real needs: 1 m/s of closing 5 cm past the bumper needs 10 m/s², and a 10 cm floor halves it.
_MAX_DECEL_MPS2 = 100.0
_FAST_LEFT_TOP = 3    # segments for Stage 1's open gap, FRONT_LEFT at highway speed

# Values from Waymo's docstrings, as report keys. Anything else is counted as other, so no
# string read from a file can reach the report.
_TIMES_OF_DAY = ("day", "dawn_dusk", "night")
_WEATHER = ("sunny", "rain")
# Each of a spread's five numbers can be one event's own value: from 5 events they'd be the events, which stay
# on PACE. From 10, at most half of them can show.
_MIN_SPREAD = 10
_QUANTILES = {"min": 0.0, "p25": 0.25, "median": 0.5, "p75": 0.75, "max": 1.0}

_FRONT, _FRONT_LEFT = CAMERA_IDS["FRONT"], CAMERA_IDS["FRONT_LEFT"]
_PAGES = Path("review", "cutins")  # under $WOD_HOME. Anything else in review is Andrew's, and stays.
_REVIEW_FPS = 2.0
_REVIEW_BEFORE_S, _REVIEW_AFTER_S = 1.0, 2.0  # from onset minus 1 s to entry plus 2 s
_REVIEW_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Cut-in review</title>
<style>
  body { margin: 16px; font-family: sans-serif; }
  .frames { display: flex; gap: 8px; overflow-x: auto; }
  figure { flex: 0 0 640px; margin: 0; }
  img { width: 100%; }
</style>
"""


@dataclass(frozen=True)
class ScanConfig:
    clear_m: float = 2.5
    inside_m: float = 1.0
    max_crossing_s: float = 6.0
    min_stay_s: float = 1.0
    stay_m: float = 1.5
    max_gap_m: float = 30.0
    max_heading_rad: float = 0.5
    min_object_speed_mps: float = 2.0
    min_ego_speed_mps: float = 3.0
    front_camera_x_m: float = 1.5
    front_half_fov_deg: float = 25.0
    path_end_allowance_m: float = 8.0
    braking_window_s: float = 3.0
    own_move_lead_s: float = 2.0
    min_own_move_m: float = 1.0
    max_lateral_change_mps2: float = 0.5
    steer_window_end_s: float = 0.5
    fast_ego_speed_mps: float = 20.0


def load_scan_config(path) -> ScanConfig:
    """configs/scan.toml. A misspelled key raises, rather than leave its default in place unnoticed."""
    return ScanConfig(**tomllib.loads(Path(path).read_text())["scan"])


def _typed(rows, types: dict) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(types)).astype(types)


def _crossings(lateral: np.ndarray, clear_m: float, inside_m: float) -> list[tuple[int, int]]:
    """(onset, entry) positions in one track: each first sample in the path after the last one clear of it.

    The last clear sample, not the first: a car can sit in the next lane for 15 s before it moves.
    An entry uses up its onset, so a car has to clear the path again to be counted again.
    """
    pairs, onset = [], None
    for i, offset in enumerate(np.abs(lateral)):
        if offset >= clear_m:
            onset = i
        elif offset <= inside_m and onset is not None:
            pairs.append((onset, i))
            onset = None
    return pairs


def _bearing_deg(boxes: pd.DataFrame, config: ScanConfig) -> pd.Series:
    """Each box's bearing, left positive, from about where FRONT sits ahead of the rear axle."""
    return np.degrees(np.arctan2(boxes["center_y_m"], boxes["center_x_m"] - config.front_camera_x_m))


def _could_enter(vehicles: pd.DataFrame, config: ScanConfig, path_end_m: float) -> np.ndarray:
    """Whether each sample could be an entry. NaN fails every test, as an unknown should.

    The path runs from the ego's first pose to path_end_allowance_m past its last, along the
    straight extension there. Without the allowance the ego would have to drive past where the
    car entered, which blinds a segment's last seconds and misses a cut-in that makes the ego
    stop. The bearing is from about where FRONT sits, and its half field of view is about 25
    degrees: close enough to need no calibration.
    """
    gap = vehicles["front_gap_m"]
    return ((gap > 0.0) & (gap <= config.max_gap_m)
            & (vehicles["heading_rad"].abs() <= config.max_heading_rad)
            & (vehicles["speed_mps"] >= config.min_object_speed_mps)
            & (vehicles["ego_speed_mps"] >= config.min_ego_speed_mps)
            & vehicles["path_s_m"].between(0.0, path_end_m + config.path_end_allowance_m)
            & (_bearing_deg(vehicles, config).abs() <= config.front_half_fov_deg)).to_numpy()


def _path_yaw(ego: pd.DataFrame, s) -> np.ndarray:
    """The way the ego's path runs at each s: the ego's yaw where it passed, and its end yaws past its ends.

    Unwrapped first, in time as the ego table runs, so a heading through ±pi doesn't read as a U-turn.
    np.interp needs s sorted, and a stop's pose noise can step it back.
    """
    driven = ego.assign(yaw_rad=np.unwrap(ego["yaw_rad"])).sort_values("path_s_m")
    return np.interp(s, driven["path_s_m"], driven["yaw_rad"])


def _from_arc(x, y, curvature):
    """Signed distance, left positive, from the circle of this curvature that leaves the origin along +x.

    The radius less the distance to the circle's center, rationalized and times the curvature, so
    a curvature of 0 gives y, a straight line, and nothing divides by a small number.
    """
    return (2.0 * y - curvature * (x * x + y * y)) / (1.0 + np.hypot(curvature * x, 1.0 - curvature * y))


def _curvature(ego: pd.DataFrame) -> np.ndarray:
    """The circle each ego row is driving: yaw rate over speed, capped, or a straight line when too slow to tell."""
    curvature = np.clip(ego["yaw_rate_radps"] / ego["speed_mps"], -_MAX_CURVATURE, _MAX_CURVATURE)
    return np.where(ego["speed_mps"] >= _MIN_TURN_SPEED_MPS, curvature, 0.0)


def _own_move(onset: pd.Series, entry: pd.Series, ego: pd.DataFrame, config: ScanConfig) -> tuple[float, float]:
    """The event's own_move_m and lateral_change_mps2: how much nearer the car came to the ego's line by its own
    move, from onset to entry, and how hard the ego pulled off that line before onset.

    lateral_offset_m is against the ego's whole driven path, so when the ego later moves into a
    car's lane, that offset falls to 0 though the car never moved. Here the line is the circle the
    ego was driving own_move_lead_s before onset, seen from that pose. A car that holds its lane
    scores about 0, and a cut-in about clear_m - inside_m. The circle stands for the lane only while
    the ego keeps driving it. Leaving it pulls the ego sideways by its curvature's change times its
    speed squared, and the largest pull from the pose to steer_window_end_s before onset is the
    second value. Past max_lateral_change_mps2 the ego may have steered, out to pass, over a lane or
    where the road bends, so _rank doubts the own move. Lane-keeping wander stays under that, but a
    jolt or a bad pose may not. The own move is NaN only when the circle or a position is unknown.
    """
    times, curvature, speed = ego[TIME].to_numpy(), _curvature(ego), ego["speed_mps"].to_numpy()
    lead = np.abs(times - (onset[TIME] - config.own_move_lead_s * 1e6)).argmin()
    end = max(onset[TIME] - config.steer_window_end_s * 1e6, times[lead])  # never before the pose, so never empty
    window = (times >= times[lead]) & (times <= end)  # not to onset: a planner may react to the car's first drift
    change = (np.abs(curvature[window] - curvature[lead]) * speed[window] ** 2).max()  # m/s², sideways
    pose = ego.iloc[lead]
    cos, sin = np.cos(pose["yaw_rad"]), np.sin(pose["yaw_rad"])
    dx = np.array([onset["pos_local_x_m"], entry["pos_local_x_m"]]) - pose["pos_local_x_m"]
    dy = np.array([onset["pos_local_y_m"], entry["pos_local_y_m"]]) - pose["pos_local_y_m"]
    at_onset, at_entry = np.abs(_from_arc(cos * dx + sin * dy, cos * dy - sin * dx, curvature[lead]))  # pose's frame
    return at_onset - at_entry, change


def _required_decel(window: pd.DataFrame, entry_time, config: ScanConfig) -> float:
    """The hardest braking the ego would have needed to avoid the car: closing² / (2 gap) at its largest.

    Over the car's samples in the window that are ahead with the gap closing. Before entry the car is
    still crossing, so only from entry on must it be in the path. A gap that never closes needs 0, and
    noise around a steady gap needs about 0: no cliff between them, as a time to collision had.
    """
    closing, gap = -window["v_rel_ego_x_mps"], window["front_gap_m"]
    counted = ((gap > 0.0) & (closing > 0.0)
               & ((window[TIME] < entry_time) | (window["lateral_offset_m"].abs() <= config.stay_m)))
    return min((closing * closing / (2.0 * gap)).where(counted, 0.0).max(), _MAX_DECEL_MPS2)


def _event(crossing: pd.DataFrame, track: pd.DataFrame, ego: pd.DataFrame, path_end_m: float,
           config: ScanConfig) -> tuple:
    """An event's row, in _EVENT_TYPES's order, from its track's samples from onset to entry, and the whole track.

    Severity counts from onset: a planner brakes as soon as it sees the car start over, so by entry its
    gap can be opening, and then the braking itself is the severity. An unknown braking counts as none.
    """
    onset, entry = crossing.iloc[0], crossing.iloc[-1]
    span = (onset[TIME], entry[TIME] + config.braking_window_s * 1e6)
    required = _required_decel(track[track[TIME].between(*span)], entry[TIME], config)
    braking = ego.loc[ego[TIME].between(*span), "a_long_mps2"].min()
    return (entry[OBJECT_ID], entry["alias"], onset[TIME], entry[TIME], onset["t_s"], entry["t_s"],
            entry["front_gap_m"] / entry["ego_speed_mps"],
            -entry["v_rel_ego_x_mps"],
            required, braking, max(required, -np.nan_to_num(braking)),  # required is never below 0, so neither is this
            (entry[TIME] - onset[TIME]) / 1e6,
            (crossing["lateral_offset_m"].diff().abs() / crossing["t_s"].diff()).max(),
            *_own_move(onset, entry, ego, config),
            entry["path_s_m"] > path_end_m)


def find_cut_ins(tracks: pd.DataFrame, ego: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    """One row per cut-in event, measured at its entry. Pure, and on Stage 1's own table shapes.

    So it reads a Stage 1 segment as well as the scan's light tables. Time limits compare whole
    microseconds, so a sample exactly at a limit is inside it. The stay allows stay_m, wider than
    inside_m: a label's noise at the edge of the path would otherwise end it a sample after entry.
    own_move_lead_s before onset, a car about to change lanes is driving along the road. One turning
    in from a side street is across it, and one pulling out of a parking spot is standing still.
    That heading is against the path where the car is, not the ego's nose, so neither a curve nor
    the ego's own turn counts against it.
    """
    frames = ego.set_index(TIME)
    path_end_m = ego.loc[ego[TIME].idxmax(), "path_s_m"]  # where the ego's last pose sits
    vehicles = tracks[tracks["type"] == "vehicle"].sort_values([OBJECT_ID, TIME], ignore_index=True)
    vehicles = vehicles.assign(ego_speed_mps=vehicles[TIME].map(frames["speed_mps"]),
                               path_s_m=vehicles["along_path_m"] + vehicles[TIME].map(frames["path_s_m"]))
    could_enter = _could_enter(vehicles, config, path_end_m)
    driving = ((np.abs(wrap_angle(vehicles["yaw_world_rad"] - _path_yaw(ego, vehicles["path_s_m"])))
                <= config.max_heading_rad) & (vehicles["speed_mps"] >= config.min_object_speed_mps)).to_numpy()
    ts, lateral = vehicles[TIME].to_numpy(), vehicles["lateral_offset_m"].to_numpy()
    events = []
    for rows in vehicles.groupby(OBJECT_ID).indices.values():  # one track's positions, in time order
        track_ts, track_lateral = ts[rows], lateral[rows]
        for onset, entry in _crossings(track_lateral, config.clear_m, config.inside_m):
            before = np.abs(track_ts - (track_ts[onset] - config.own_move_lead_s * 1e6)).argmin()  # _own_move's time
            # Crossing traffic passes straight through. A track that ends inside has stayed as long as it could,
            # but a hole in it hasn't: the stay runs to the first sample at or past min_stay_s.
            stay = track_lateral[entry:np.searchsorted(track_ts, track_ts[entry] + config.min_stay_s * 1e6) + 1]
            if (could_enter[rows[entry]] and driving[rows[before]]
                    and track_ts[entry] - track_ts[onset] <= config.max_crossing_s * 1e6
                    and (np.abs(stay) <= config.stay_m).all()):
                track = vehicles.iloc[rows]
                events.append(_event(track.iloc[onset:entry + 1], track, ego, path_end_m, config))
    return _typed(events, _EVENT_TYPES).sort_values(["entry_timestamp_micros", OBJECT_ID], ignore_index=True)


def _no_camera_rows(boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synced and projected tables with no rows, since the scan reads no camera component.

    build_tracks_table joins them on time and object, so those keep the boxes' types.
    """
    keys = boxes[[TIME, OBJECT_ID]].iloc[:0]
    synced = keys.assign(most_visible_camera=np.int8(0), center_x=0.0, center_y=0.0, center_z=0.0,
                         size_x=0.0, size_y=0.0, size_z=0.0, heading=0.0)
    return synced, keys.assign(camera=np.int8(0))


def _conditions(root, ref: SegmentRef) -> dict:
    """time_of_day, weather and location from stats' first row in time. Each holds for the whole segment."""
    table = pq.read_table(component_path(root, ref, STATS), columns=[TIMESTAMP, *(col(STATS, c) for c in _CONDITIONS)])
    first = table.sort_by(TIMESTAMP).slice(0, 1).to_pylist()[0]
    return {name: first[col(STATS, name)] for name in _CONDITIONS}


def _front_left(root, ref: SegmentRef) -> CameraModel | None:
    """FRONT_LEFT's calibration, or None if the segment has none it can read. That costs only its left_at_speed."""
    try:
        return load_cameras(root, ref, [_FRONT_LEFT])[_FRONT_LEFT]
    except Exception:  # missing, empty, cut short, or no FRONT_LEFT row Stage 1 can use: never the segment's cut-ins
        return None


def _left_at_speed(tracks: pd.DataFrame, ego: pd.DataFrame, cam: CameraModel | None, margin_px: float,
                   config: ScanConfig) -> int:
    """Vehicle boxes FRONT_LEFT shows whole while the ego is fast: what Stage 1's check 3 can measure there.

    For picking segments that close Stage 1's open gap, FRONT_LEFT at highway speed. Each box goes
    through the segment's own calibration as check 3's frame_pose reading does, and counts if all
    eight corners fall margin_px inside the image, by check 3's own test. Check 3 can't measure a box
    the border cuts off, and a car in the next lane is usually too close to fit.
    """
    if cam is None:
        return 0
    vehicles = tracks[tracks["type"] == "vehicle"]
    fast = vehicles[vehicles[TIME].map(ego.set_index(TIME)["speed_mps"]) >= config.fast_ego_speed_mps]
    camera_from_vehicle = cam.camera_from_vehicle  # a property that inverts the extrinsic: once here, not per box
    rectangles = (box_rectangle(transform_points(camera_from_vehicle, row_corners(box)), cam)
                  for box in fast.itertuples())
    return sum(rectangle is not None and _inside(rectangle, cam, margin_px) for rectangle in rectangles)


def scan_segment(root, ref, stage1: Stage1Config, config: ScanConfig) -> tuple[pd.DataFrame, dict]:
    """A segment's events and its row of segments.parquet, from labels, poses, stats and calibration alone.

    Without FRONT_LEFT's calibration, a segment keeps its events but counts nothing for left_at_speed.
    """
    poses = load_poses(root, ref)
    boxes = load_lidar_boxes(root, ref)
    ego, path = build_ego_table(poses, stage1.motion)
    tracks = build_tracks_table(boxes, *_no_camera_rows(boxes), poses, ego, path, stage1.motion,
                                stage1.rear_axle_to_front_m)
    events = find_cut_ins(tracks, ego, config)
    front_left = _front_left(root, ref)
    summary = {
        "split": ref.split,
        "context_name": ref.context_name,
        "n_frames": len(ego),
        "ego_speed_median_mps": ego["speed_mps"].median(),
        "ego_speed_max_mps": ego["speed_mps"].max(),
        "vehicle_tracks": tracks.loc[tracks["type"] == "vehicle", OBJECT_ID].nunique(),
        "events": len(events),
        "best_headway_s": events["headway_s"].min(),
        "left_at_speed": _left_at_speed(tracks, ego, front_left, stage1.checks.border_margin_px, config),
        "front_left_calibrated": front_left is not None,
        **_conditions(root, ref),
    }
    events.insert(0, "split", ref.split)
    events.insert(1, "context_name", ref.context_name)
    return events, summary


def _scan_or_fail(ref: SegmentRef, stage1: Stage1Config, config: ScanConfig) -> tuple[pd.DataFrame, dict] | str:
    """scan_segment's result, or the name of what it raised. Never the message: it can echo Dataset values."""
    try:
        return scan_segment(stage1.root, ref, stage1, config)
    except Exception as exc:
        return type(exc).__name__


def _scan_all(refs: list[SegmentRef], stage1: Stage1Config, config: ScanConfig, workers: int) -> list:
    """Each segment's result or failure, in order. A pool takes one segment per task."""
    scan = partial(_scan_or_fail, stage1=stage1, config=config)
    if workers == 1:  # no pool to start, and a failure is easier to follow in one process
        return [scan(ref) for ref in refs]
    # Linux forks workers, and they'd keep main's handler: a SIGTERM would be a SystemExit they catch and carry
    # on from. A pool broken by one worker's crash would then wait on the rest forever, and at Slurm's time
    # limit main would wait for them to finish every segment. With the default, they just die.
    with ProcessPoolExecutor(workers, initializer=signal.signal, initargs=(signal.SIGTERM, signal.SIG_DFL)) as pool:
        return list(pool.map(scan, refs))


def _rank(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    """Segments by their most severe event, for a top 10 checked by eye, where a non-event costs more than a miss.

    An event goes last when its car barely moved itself, or the ego pulled sideways past
    max_lateral_change_mps2 before onset: the ego probably moved instead. Then the largest
    severity, so braking done and braking needed rank on one scale, then the smallest headway.
    Names break any tie that's left, so a rerun lists the same segments.
    """
    ego_moved = ~((events["own_move_m"] >= config.min_own_move_m)
                  & (events["lateral_change_mps2"] <= config.max_lateral_change_mps2))  # an unknown goes last too
    severest_first = events.assign(ego_moved=ego_moved).sort_values(
        ["ego_moved", "severity_mps2", "headway_s", *_SEGMENT_KEYS], ascending=[True, False, True, True, True])
    return severest_first.drop_duplicates(_SEGMENT_KEYS)[_SEGMENT_KEYS]


def _fast_left(segments: pd.DataFrame) -> pd.DataFrame:
    """The few segments with the most vehicle samples about FRONT_LEFT at speed, if any. Names break ties."""
    busy = segments[segments["left_at_speed"] > 0]
    return busy.sort_values(["left_at_speed", *_SEGMENT_KEYS], ascending=[False, True, True]).head(_FAST_LEFT_TOP)


def _spread(values: pd.Series) -> dict:
    """The count, then the five-number summary once there are enough values that it isn't them."""
    values = values.dropna().astype(np.float64)
    if len(values) < _MIN_SPREAD:
        return {"count": len(values)}
    return {"count": len(values), **{name: values.quantile(q) for name, q in _QUANTILES.items()}}


def _category(value, known: tuple[str, ...]) -> str:
    """Dawn/Dusk as dawn_dusk, if it's a value Waymo documents, and otherwise other."""
    key = re.sub(r"[^a-z]+", "_", str(value).lower())
    return key if key in known else "other"


def _snake(name: str) -> str:
    """ArrowInvalid as arrow_invalid, since report keys are lowercase. A type's name is the code's, not the data's."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _scan(args, stage1: Stage1Config, config: ScanConfig, home: Path, report: Report) -> None:
    """Scan, rank, and write. Each step is guarded, so the report says how far a run got."""
    with guarded(report, "scan"):
        report.add("config", **asdict(config))
        refs = [SegmentRef(split, path.stem) for split in args.splits
                for path in sorted((stage1.root / split / LIDAR_BOX).glob("*.parquet"))][:args.limit]
        frames, summaries, failures = [], [], Counter()
        for result in _scan_all(refs, stage1, config, args.workers):
            if isinstance(result, str):
                failures[result] += 1
            else:
                frames.append(result[0])
                summaries.append(result[1])
        # The empty frame sets the columns if every segment failed.
        events = pd.concat([_typed([], _SCANNED_EVENT_TYPES), *frames], ignore_index=True)
        segments = pd.DataFrame(summaries, columns=_SEGMENT_COLUMNS)
        ranked = _rank(events, config).head(args.top)
        unsteered = _rank(events.assign(lateral_change_mps2=0.0), config).head(args.top)  # as if the ego never steered
        top = pd.concat([ranked, unsteered]).drop_duplicates().merge(segments, on=_SEGMENT_KEYS)  # each segment once
        fast_left = _fast_left(segments)
        with_events = int((segments["events"] > 0).sum())
        report.add("segments", scanned=len(segments), failed=failures.total(),
                   failed_by_type={_snake(name): count for name, count in failures.items()},
                   with_left_at_speed=int((segments["left_at_speed"] > 0).sum()),
                   without_calibration=len(segments) - int(segments["front_left_calibrated"].sum()))
        steered = ~(events["lateral_change_mps2"] <= config.max_lateral_change_mps2)  # an unknown pull too
        report.add("events", events=len(events), segments_with_events=with_events,
                   car_moved=int((~steered & (events["own_move_m"] >= config.min_own_move_m)).sum()),
                   ego_steered=int(steered.sum()),
                   past_path_end=int(events["past_path_end"].sum()),
                   **{name: _spread(events[name])
                      for name in ("headway_s", "required_decel_mps2", "ego_min_a_long_mps2", "severity_mps2",
                                   "crossing_s", "lateral_change_mps2")})
        report.add("conditions", **{name: Counter(_category(value, known) for value in segments[name])
                                    for name, known in (("time_of_day", _TIMES_OF_DAY), ("weather", _WEATHER))})
        report.add("top", segments=len(ranked), steering_test_off=len(top) - len(ranked),
                   daytime=sum(_category(value, _TIMES_OF_DAY) == "day" for value in top["time_of_day"]))

    with guarded(report, "outputs"):
        directory = stage1.outputs / "scan"
        directory.mkdir(parents=True, exist_ok=True)
        events.to_parquet(directory / "events.parquet", index=False)
        segments.to_parquet(directory / "segments.parquet", index=False)
        # Context lists, so mode 600: main's umask sees to that before anything is written.
        for listing, chosen in (("cutin_top.txt", top), ("fast_left.txt", fast_left)):
            names = "".join(f"{name}\n" for name in chosen["context_name"])
            write_atomic(home / "selected" / listing, names.encode())
        shutil.rmtree(home / _PAGES, ignore_errors=True)  # pages go by rank, and the ranks just changed
        print(f"segments: {len(segments)} scanned, {failures.total()} failed; events: {len(events)} in {with_events} "
              f"segments; top: {len(ranked)}, steering test off: {len(top) - len(ranked)}; fast left: {len(fast_left)}")


def _figure(jpeg: bytes, caption: str) -> str:
    return (f'<figure><img alt="" src="data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _event_line(event, config: ScanConfig) -> str:
    """The event's numbers, and plainly when the ego, rather than the car, may have moved, or the pull is unknown."""
    braked = max(0.0, -np.nan_to_num(event.ego_min_a_long_mps2))  # as severity counts it
    own_move = "unknown" if np.isnan(event.own_move_m) else f"{event.own_move_m:.2f} m"
    unknown_pull = ", pull before onset unknown" if np.isnan(event.lateral_change_mps2) else ""  # ranks last, no steer
    small = ". The ego probably moved, not the car" if event.own_move_m < config.min_own_move_m else ""
    pulled = (f". The ego pulled {event.lateral_change_mps2:.2f} m/s² sideways before onset, so the own move may be"
              " wrong" if event.lateral_change_mps2 > config.max_lateral_change_mps2 else "")
    return (f"{event.alias}: headway {event.headway_s:.2f} s, "
            f"needed {event.required_decel_mps2:.1f} m/s² to avoid it, braked {braked:.1f} m/s², "
            f"crossing {event.crossing_s:.2f} s, own move {own_move}{unknown_pull}{small}{pulled}")


def _review_page(segment: SegmentOutputs, events: pd.DataFrame, stage1: Stage1Config, config: ScanConfig,
                 rank: int) -> str:
    """FRONT at 2 FPS around each event, with the cut-in vehicle drawn thick and the event's numbers above.

    Drawn here and never cached: a cached image is shared, so it can't single anyone out.
    """
    ref = SegmentRef(segment.meta["split"], segment.meta["context_name"])
    cam = camera_models(segment.meta)[_FRONT]
    front = segment.frames.loc[segment.frames["camera"] == _FRONT, TIME].to_numpy()
    t0 = segment.meta["t0_micros"]
    parts = [_REVIEW_HEAD, f"<h1>Cut-in candidate {rank}</h1>\n"]
    if events.empty:
        parts.append("<p>No events in Stage 1's tables.</p>\n")
    for event in events.itertuples():
        marks = {event.onset_timestamp_micros: ", onset", event.entry_timestamp_micros: ", entry"}
        # Only frames FRONT has: a camera can lose one that the labels kept.
        shown = select_frames(front, t0, _REVIEW_FPS, event.onset_t_s - _REVIEW_BEFORE_S,
                              event.entry_t_s + _REVIEW_AFTER_S, closed_end=True,
                              force_include=np.intersect1d(list(marks), front)).timestamps
        jpegs = read_jpegs(stage1.root, ref, _FRONT, shown)
        parts.append(f"<h2>{html.escape(_event_line(event, config))}</h2>\n")
        figures = [_figure(render_annotated(jpegs[ts], boxes_for_image(segment, ts, _FRONT, cam), cam,
                                            stage1.profiles["full"], event.alias),
                           f"{event.alias} at {(ts - t0) / 1e6:.2f} s{marks.get(ts, '')}") for ts in shown]
        parts.append(f'<div class="frames">{"".join(figures)}</div>\n')
    return "".join(parts)


def _review(stage1: Stage1Config, config: ScanConfig, home: Path) -> None:
    """A page per listed segment that agatha-stage1 has run on, named by rank, so no page names its context.

    Events come again from Stage 1's own tables, which give the scan's events. A page that fails
    is only counted: its message can echo Dataset values.
    """
    contexts = (home / "selected" / "cutin_top.txt").read_text().split()
    shutil.rmtree(home / _PAGES, ignore_errors=True)  # a page left from an older list would show another segment
    counts = Counter(pages=0, events=0, without_stage1=0, failed=0)
    for rank, context in enumerate(contexts, start=1):
        directory = stage1.outputs / context
        if not (directory / "segment.json").exists():  # Stage 1 writes it last, so the tables are whole
            counts["without_stage1"] += 1
            continue
        try:
            segment = load_segment(directory)
            events = find_cut_ins(segment.tracks, segment.ego, config)
            write_atomic(home / _PAGES / f"cutin_{rank}.html",
                         _review_page(segment, events, stage1, config, rank).encode())
            counts.update(pages=1, events=len(events))
        except Exception:
            counts["failed"] += 1
    print(f"review: {counts['pages']} pages, {counts['events']} events; "
          f"{counts['without_stage1']} without Stage 1 outputs, {counts['failed']} failed")
    print(f"pages: {home / _PAGES}")


def main(argv=None) -> int:
    os.umask(0o077)  # before anything is written: the tables and the top list name contexts
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # Slurm's time limit: `guarded` and the finally still run
    # For what no step guards, like a bad config: a traceback prints paths, and those can hold a context.
    sys.excepthook = lambda *_: print("agatha-scan: failed outside any step", file=sys.stderr)
    parser = _QuietParser(prog="agatha-scan", description="Rank labeled segments by how likely each holds a cut-in.")
    parser.add_argument("command", nargs="?", choices=["scan", "review"], default="scan",
                        help="review: a page per listed segment, once agatha-stage1 has run on it")
    parser.add_argument("--config", default="configs/stage1.toml")
    parser.add_argument("--scan-config", default="configs/scan.toml")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--workers", type=int, default=1, help="processes, one segment per task")
    parser.add_argument("--top", type=int, default=10, metavar="K",
                        help="list the ranked top K, then up to K more that rank there with the steering test off")
    parser.add_argument("--limit", type=int, help="scan only the first N segments, to try a run")
    args = parser.parse_args(argv)

    stage1, config = load_config(args.config), load_scan_config(args.scan_config)
    home = Path(os.environ["WOD_HOME"])  # env.sh sets it. The list and pages sit beside the data, not in the outputs.
    if args.command == "review":
        _review(stage1, config, home)
        return 0

    # Report takes a context. The scan spans many, so it passes the word scan, which Report hashes like any name.
    report = Report("scan", stage1.preprocessing_version)
    path = stage1.outputs / "reports" / "scan.json"
    try:
        with guarded(report, "start"):
            path.unlink(missing_ok=True)  # no report then means no finished run
        _scan(args, stage1, config, home, report)
    finally:
        with guarded(report, "report"):
            report.write(path)
            print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import base64
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agatha import scan
from agatha.columns import (
    CAMERA_CALIBRATION, CAMERA_IDS, LIDAR_BOX, OBJECT, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, STATS, TIMESTAMP,
    VEHICLE_POSE, col,
)
from agatha.config import RenderProfile, load_config
from agatha.geometry import CameraModel, box_corners, box_rectangle, rot_z, se3, transform_points, wrap_angle
from agatha.loader import SegmentRef, component_path, load_poses, read_jpegs
from agatha.render import DrawBox, boxes_for_image, render_annotated
from agatha.report import VOCABULARY, Report, context_label
from agatha.run_stage1 import process_segment
from agatha.scan import ScanConfig, find_cut_ins, load_scan_config, scan_segment
from agatha.store import camera_models, load_segment
from agatha.synthetic import LABEL_OFFSET_S, OBJECT_IDS, _camera_model, write_synthetic_segment
from agatha.tracks import build_ego_table

REPO = Path(__file__).parents[1]
CONFIG_PATH = REPO / "configs" / "stage1.toml"
SCAN_CONFIG_PATH = REPO / "configs" / "scan.toml"
CONFIG_ARGS = ["--config", str(CONFIG_PATH), "--scan-config", str(SCAN_CONFIG_PATH)]
CONFIG = ScanConfig()
FRONT = CAMERA_IDS["FRONT"]
TIME = "frame_timestamp_micros"
T0 = 1_550_000_000_000_000
CONTEXT = "synthetic_cut_in"
# Shaped like real names, so a report would hash them, and a leak can't hide behind a public name.
PRIVATE = {name: f"{str(digit) * 20}_100_000_120_000"
           for digit, name in enumerate(("near", "far", "quiet", "broken", "busy", "light", "right_only",
                                         "uncalibrated"), start=1)}
CONDITIONS = ("Day", "location_sf", "sunny")  # time_of_day, location, weather: REQUIRED_COLUMNS[STATS]'s order
VEHICLE = (4.6, 1.9, 1.6)
LANE = 3.7
TEN_HZ = np.arange(200) / 10.0  # a segment's 19.9 s
# FRONT_LEFT as synthetic.py builds it: FRONT's real lens, from Waymo's camera_model_test.cc, turned 45 degrees left.
FRONT_LEFT_CAM = _camera_model("FRONT_LEFT")

# The spec's tables, column for column.
EVENT_COLUMNS = """
    split context_name laser_object_id alias onset_timestamp_micros entry_timestamp_micros onset_t_s entry_t_s
    headway_s closing_mps required_decel_mps2 ego_min_a_long_mps2 severity_mps2 crossing_s peak_lateral_mps
    own_move_m lateral_change_mps2 past_path_end""".split()
SEGMENT_COLUMNS = """
    split context_name n_frames ego_speed_median_mps ego_speed_max_mps vehicle_tracks events best_headway_s
    left_at_speed front_left_calibrated time_of_day weather location""".split()


@pytest.fixture(autouse=True)
def process_state_is_restored():
    """main takes over the process it runs in, and in these tests that's pytest's."""
    handler, hook, umask = signal.getsignal(signal.SIGTERM), sys.excepthook, os.umask(0o077)
    yield
    signal.signal(signal.SIGTERM, handler)
    sys.excepthook = hook
    os.umask(umask)


@pytest.fixture(scope="module")
def stage1(tmp_path_factory):
    """Stage 1's config, for its motion settings and ego length. Its paths go unused."""
    unused = tmp_path_factory.mktemp("unused")
    return load_config(CONFIG_PATH, root=unused, outputs=unused)


def write_rows(root, ref: SegmentRef, tag: str, rows: list[tuple]) -> None:
    """A component in the real layout, columns as columns.py lists them. Reversed, so file order isn't time order."""
    names = [*REQUIRED_COLUMNS[tag], *OPTIONAL_COLUMNS.get(tag, [])]
    path = component_path(root, ref, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(dict(zip(names, map(list, zip(*rows[::-1]))))), path)


def write_stats(root, ref: SegmentRef, conditions=CONDITIONS, first=None) -> None:
    """One stats row per pose, each holding `conditions`, except the earliest, which holds `first` if given."""
    times = sorted(pq.read_table(component_path(root, ref, VEHICLE_POSE), columns=[TIMESTAMP])[TIMESTAMP].to_pylist())
    write_rows(root, ref, STATS, [(ref.context_name, ts, *(first if first and ts == times[0] else conditions))
                                  for ts in times])


def write_calibration(root, ref: SegmentRef, cameras=("FRONT", "FRONT_LEFT")) -> None:
    """A camera_calibration file, a row per camera as synthetic.py builds them, as the label download fetches it."""
    write_rows(root, ref, CAMERA_CALIBRATION, [
        (ref.context_name, CAMERA_IDS[camera], *cam.intrinsic, cam.extrinsic.ravel().tolist(), cam.width, cam.height,
         cam.rolling_shutter_direction) for camera, cam in ((camera, _camera_model(camera)) for camera in cameras)])


def write_unreadable_calibration(root, ref: SegmentRef, how: str) -> None:
    """A camera_calibration file Stage 1 can't read: empty, cut short, or with no width in FRONT_LEFT's row."""
    write_calibration(root, ref)
    path = component_path(root, ref, CAMERA_CALIBRATION)
    if how == "empty":
        path.write_bytes(b"")
    elif how == "cut_short":
        path.write_bytes(path.read_bytes()[:path.stat().st_size // 2])
    else:
        table, width = pq.read_table(path), col(CAMERA_CALIBRATION, "width")
        pq.write_table(table.set_column(table.schema.get_field_index(width), width,
                                        pa.nulls(len(table), table.schema.field(width).type)), path)


def smoothstep(t, start: float, duration: float) -> np.ndarray:
    """0 until start, 1 from start + duration, and an S between: the shape of a lane change."""
    x = np.clip((np.asarray(t, dtype=np.float64) - start) / duration, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cuts_in(start: float, duration: float) -> np.ndarray:
    """Lateral offset at 10 Hz of a car moving in from the next lane on the right, from start, over duration."""
    return -LANE * (1.0 - smoothstep(TEN_HZ, start, duration))


def braking(speed: float, start: float, decel: float, until: float) -> np.ndarray:
    """Distance at 10 Hz of something at `speed` that brakes at `decel` from `start` until it's down to `until`."""
    slowing = np.clip(TEN_HZ - start, 0.0, (speed - until) / decel)
    return (speed * np.minimum(TEN_HZ, start) + speed * slowing - decel / 2.0 * slowing**2
            + until * np.clip(TEN_HZ - start - (speed - until) / decel, 0.0, None))


def on_road(s, d, curvature: float, straight_m=0.0) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) s meters along a road and d meters left of it.

    The road leaves the origin along +x, runs straight for straight_m, then bends left at a constant curvature, or
    right if that's negative.
    """
    s, d = np.broadcast_arrays(np.asarray(s, dtype=np.float64), np.asarray(d, dtype=np.float64))
    if not curvature:
        return s, d
    radius, straight = 1.0 / curvature, np.minimum(s, straight_m) if straight_m else 0.0
    turned = (s - straight) / radius
    return straight + (radius - d) * np.sin(turned), radius - (radius - d) * np.cos(turned)


def driving(s, d=0.0, curvature=0.0, straight_m=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World (x, y, yaw) at 10 Hz of something s(t) along a road and d(t) left of it, facing the way it moves."""
    x, y = on_road(s, d, curvature, straight_m)
    vx, vy = np.gradient(x, 0.1), np.gradient(y, 0.1)
    bent = np.asarray(s) - (np.minimum(s, straight_m) if straight_m else 0.0)  # how far round, for a car standing still
    return x, y, np.where(np.hypot(vx, vy) > 0.1, np.arctan2(vy, vx), curvature * bent)


def turning_in(end_s: float, radius=10.0, speed=6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World (x, y, yaw) at 10 Hz of something that comes up a side street on the right of a road along x, turns
    right onto it round a quarter circle, and leaves the turn end_s m along the road at 9 s, all at `speed`."""
    u = speed * (TEN_HZ - 9.0) + np.pi * radius / 2.0  # along its own way, from the start of the turn
    turned = np.clip(u / radius, 0.0, np.pi / 2.0)
    x = end_s - radius * np.cos(turned) + np.clip(u - np.pi * radius / 2.0, 0.0, None)
    return x, np.minimum(u, 0.0) - radius * (1.0 - np.sin(turned)), np.pi / 2.0 - turned


def turning_onto(u, d, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World (x, y, yaw) at 10 Hz of something u(t) m along a way and d(t) m left of it. The way comes up a side street
    on the right of a road along x, and turns right onto it round a quarter circle from u = 0, leaving it at x = 0."""
    u, d = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(d, dtype=np.float64))
    turned = np.clip(u / radius, 0.0, np.pi / 2.0)
    x = np.clip(u - np.pi * radius / 2.0, 0.0, None) - (radius + d) * np.cos(turned)
    y = np.minimum(u, 0.0) - radius + (radius + d) * np.sin(turned)
    return x, y, np.arctan2(np.gradient(y, 0.1), np.gradient(x, 0.1))


def write_scene(root, name: str, ego: tuple, cars: dict[str, tuple], lost=None) -> SegmentRef:
    """A scene at 10 Hz: the ego's rear axle and each car at their own world (x, y, yaw), as `driving` gives them.

    Labels go in the ego's frame, as Waymo's do, with world velocities as their speed. The frames strictly between
    the two times in `lost`, if given, are missing, poses and labels alike.
    """
    ref = SegmentRef("training", name)
    ego_x, ego_y, ego_yaw = ego
    ts = T0 + np.arange(len(ego_x)) * 100_000
    t_s = (ts - T0) / 1e6
    kept = np.flatnonzero(~((t_s > lost[0]) & (t_s < lost[1]))) if lost else range(len(ts))
    poses = [se3(rot_z(ego_yaw[k]), [ego_x[k], ego_y[k], 0.0]).ravel().tolist() for k in kept]
    write_rows(root, ref, VEHICLE_POSE, [(name, int(ts[k]), pose) for k, pose in zip(kept, poses)])
    cos, sin = np.cos(ego_yaw), np.sin(ego_yaw)
    boxes = []
    for car, (x, y, yaw) in cars.items():
        vx, vy = np.gradient(x, t_s), np.gradient(y, t_s)
        ahead, left = cos * (x - ego_x) + sin * (y - ego_y), cos * (y - ego_y) - sin * (x - ego_x)
        boxes += [(name, int(ts[k]), car, ahead[k], left[k], 0.8, *VEHICLE, wrap_angle(yaw[k] - ego_yaw[k]), 1, 500,
                   400, vx[k], vy[k], 0.0, 0.0, 0.0, 0.0) for k in kept]
    write_rows(root, ref, LIDAR_BOX, boxes)
    write_stats(root, ref)
    return ref


def scanned(root, scenes: dict[str, tuple], stage1, config=CONFIG) -> pd.DataFrame:
    """Each scene's events, from its own segment, as `write_scene` takes them."""
    return pd.concat([scan_segment(root, write_scene(root, name, *scene), stage1, config)[0]
                      for name, scene in scenes.items()], ignore_index=True)


def tables(lateral, ego_speed=10.0, **columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One car's tracks rows at 10 Hz with these lateral offsets, and the ego's rows, on a straight road along x.

    Everything else passes with room to spare unless `columns` says otherwise: 12 m ahead and closing at 2 m/s,
    nearly parallel to the ego and its path, both moving, 5 m along the path from the ego, and 4 degrees off
    FRONT's axis. The ego covers 1 m a sample, so its path ends past the car, and the car's own move is its offset's.
    """
    ts = T0 + np.arange(len(lateral)) * 100_000
    t_s = (ts - T0) / 1e6
    lateral, ego_x = np.asarray(lateral, dtype=np.float64), np.arange(len(lateral), dtype=np.float64)
    tracks = pd.DataFrame({
        TIME: ts, "t_s": t_s, "laser_object_id": "CAR", "alias": "V001", "type": "vehicle",
        "lateral_offset_m": lateral, "front_gap_m": 12.0, "heading_rad": 0.05, "yaw_world_rad": 0.05, "speed_mps": 12.0,
        "along_path_m": 5.0, "center_x_m": 17.0, "center_y_m": -1.0, "v_rel_ego_x_mps": -2.0,
        "pos_local_x_m": ego_x + 5.0, "pos_local_y_m": lateral,
    }).assign(**columns)
    ego = pd.DataFrame({TIME: ts, "t_s": t_s, "speed_mps": ego_speed, "a_long_mps2": -0.5, "path_s_m": ego_x,
                        "pos_local_x_m": ego_x, "pos_local_y_m": 0.0, "yaw_rad": 0.0, "yaw_rate_radps": 0.0})
    return tracks, ego


def lane_change(clear=-3.0, inside=0.0, settled=None, crossing_s=1.0, stay_s=2.0, tail_s=1.0, clear_s=2.0, **kwargs):
    """Clear of the path for clear_s, between for crossing_s, at `inside` at entry and at `settled` (`inside` unless
    given) through stay_s after it, then clear.

    So by default onset is at 1.9 s and entry at 2.9 s, in row 29 of 60.
    """
    between = math.copysign(1.75, clear)
    settled = inside if settled is None else settled
    lateral = ([clear] * round(clear_s * 10) + [between] * (round(crossing_s * 10) - 1) + [inside]
               + [settled] * round(stay_s * 10) + [clear] * round(tail_s * 10))
    return tables(lateral, **kwargs)


def at_bearing(degrees: float) -> dict:
    """Center columns that put a car `degrees` off FRONT's axis, from 1.5 m ahead of the rear axle."""
    return {"center_x_m": 21.5, "center_y_m": 20.0 * np.tan(np.radians(degrees))}


def scripted_time(offset_m: float) -> float:
    """When the synthetic cut-in comes within offset_m of the path: 3.7 m out at 8 s to 0 at 12 s, on a smoothstep."""
    x = np.linspace(0.0, 1.0, 100_001)
    return 8.0 + 4.0 * x[np.argmax(3.7 * (1.0 - x * x * (3.0 - 2.0 * x)) <= offset_m)]


def wod_env(root: Path, outputs: Path, home: Path) -> dict[str, str]:
    """What env.sh exports, pointed at a test's directories."""
    return {"WOD_ROOT": str(root), "WOD_OUTPUTS": str(outputs), "WOD_HOME": str(home)}


def scan_cli(*args, env: dict) -> subprocess.CompletedProcess:
    """The command a job runs, so the exit code, stdout, and stderr are the real ones. None in `env` unsets.

    The umask is the usual default, not env.sh's 077, so privacy can't come from the caller.
    """
    env = {name: value for name, value in {**os.environ, **env}.items() if value is not None}
    return subprocess.run([sys.executable, "-m", "agatha.scan", *args], capture_output=True, text=True,
                          timeout=300, env=env, umask=0o022)


def leaves(value) -> list:
    """Every string and number under a parsed JSON value."""
    if isinstance(value, dict):
        value = list(value.values())
    return [leaf for item in value for leaf in leaves(item)] if isinstance(value, list) else [value]


# The config.

def test_scan_toml_holds_the_spec_thresholds():
    assert load_scan_config(SCAN_CONFIG_PATH) == ScanConfig()


def test_scan_toml_sets_every_threshold_in_the_dataclass_order():
    """Equal to the defaults alone, a key missing from the file would pass unnoticed."""
    assert list(tomllib.loads(SCAN_CONFIG_PATH.read_text())["scan"]) == list(asdict(CONFIG))


def test_scan_toml_lines_up_its_comments():
    """Every threshold has a comment, and they all start in one column."""
    lines = [line for line in SCAN_CONFIG_PATH.read_text().splitlines() if " = " in line]
    assert len(lines) == len(asdict(CONFIG))
    assert all("#" in line for line in lines)
    assert len({line.index("#") for line in lines}) == 1


@pytest.mark.parametrize("field", list(asdict(CONFIG)))
def test_each_threshold_is_read_from_scan_toml(tmp_path, field):
    path = tmp_path / "scan.toml"
    path.write_text(re.sub(rf"^{field} = \S+", f"{field} = 123.5", SCAN_CONFIG_PATH.read_text(), flags=re.M))
    assert asdict(load_scan_config(path)) == {**asdict(CONFIG), field: 123.5}


def test_a_misspelled_threshold_fails_rather_than_keep_its_default(tmp_path):
    path = tmp_path / "scan.toml"
    path.write_text(SCAN_CONFIG_PATH.read_text().replace("clear_m =", "clear_meters ="))
    with pytest.raises(TypeError):
        load_scan_config(path)


def test_no_threshold_lives_in_stage1_toml():
    """stage1.toml is hashed into preprocessing_version, so tuning the scan there would invalidate Stage 1 outputs."""
    text = CONFIG_PATH.read_text()
    assert not [name for name in asdict(CONFIG) if name in text]


# The default synthetic segment, as the scan sees it.

@pytest.fixture(scope="module")
def synthetic_scan(synthetic_root, tmp_path_factory, stage1) -> SimpleNamespace:
    """The default synthetic segment's labels and poses, alone, plus the stats file synthetic.py doesn't write.

    The earliest stats row differs from the rest, to show which row the scan reads.
    """
    source, ref = synthetic_root
    root = tmp_path_factory.mktemp("labels")
    for tag in (VEHICLE_POSE, LIDAR_BOX):
        component_path(root, ref, tag).parent.mkdir(parents=True)
        shutil.copy(component_path(source, ref, tag), component_path(root, ref, tag))
    write_stats(root, ref, first=("Dawn/Dusk", "location_phx", "rain"))
    events, summary = scan_segment(root, ref, stage1, CONFIG)
    return SimpleNamespace(root=root, ref=ref, events=events, summary=summary)


def test_the_scan_needs_no_camera_component(synthetic_scan):
    assert sorted(path.name for path in (synthetic_scan.root / "training").iterdir()) == [LIDAR_BOX, STATS,
                                                                                           VEHICLE_POSE]


def test_the_synthetic_cut_in_is_found_near_its_scripted_time(synthetic_scan):
    [event] = synthetic_scan.events.itertuples()
    assert event.laser_object_id == OBJECT_IDS["cut_in"]
    # Labels sit LABEL_OFFSET_S after their frame. Entry is the first sample within 1 m, onset the last 2.5 m out.
    assert 0.0 <= event.entry_t_s + LABEL_OFFSET_S - scripted_time(1.0) < 0.11
    assert 0.0 <= scripted_time(2.5) - (event.onset_t_s + LABEL_OFFSET_S) < 0.11
    assert event.headway_s > 0.0


@pytest.mark.parametrize("name", [name for name in OBJECT_IDS if name != "cut_in"])
def test_no_other_synthetic_object_is_found(synthetic_scan, name):
    """The lead, the parked cars, the car beside the ego, the occluded car, the crossing pedestrian, and the sign."""
    assert OBJECT_IDS[name] not in set(synthetic_scan.events["laser_object_id"])


def test_the_synthetic_cut_ins_numbers_follow_its_script(synthetic_scan):
    event = synthetic_scan.events.iloc[0]
    t = event["entry_t_s"] + LABEL_OFFSET_S
    # 3 + 1.6 t m ahead of the rear axle, less half its length and the assumed 4 m to the bumper, at 14 - 0.4 t m/s.
    assert event["headway_s"] == pytest.approx((3.0 + 1.6 * t - 2.3 - 4.0) / (14.0 - 0.4 * t), abs=0.03)
    assert event["crossing_s"] == pytest.approx(scripted_time(1.0) - scripted_time(2.5), abs=0.11)
    assert event["peak_lateral_mps"] == pytest.approx(1.5 * 3.7 / 4.0, abs=0.02)  # the smoothstep's steepest
    assert event["ego_min_a_long_mps2"] == pytest.approx(-0.4, abs=0.03)  # the ego brakes gently throughout
    assert event["closing_mps"] < 0.0  # it's 1.6 m/s faster than the ego, so the gap opens
    assert event["required_decel_mps2"] == 0.0 and not event["past_path_end"]
    assert event["severity_mps2"] == -event["ego_min_a_long_mps2"]  # the gentle braking is all there is
    # The ego follows the road's 200 m radius bend, so the circle it drives is the road: the own move is how much
    # nearer the cut-in came to the ego's lane.
    offset = 3.7 * (1.0 - smoothstep([event["onset_t_s"], event["entry_t_s"]], 8.0 - LABEL_OFFSET_S, 4.0))
    assert event["own_move_m"] == pytest.approx(offset[0] - offset[1], abs=0.03)


def test_the_segment_row_describes_the_segment(synthetic_scan):
    summary = synthetic_scan.summary
    assert list(summary) == SEGMENT_COLUMNS
    assert (summary["split"], summary["context_name"], summary["n_frames"], summary["events"]) == (
        "training", CONTEXT, 199, 1)
    assert summary["vehicle_tracks"] == 7  # every actor but the pedestrian and the sign
    assert summary["ego_speed_max_mps"] == pytest.approx(14.0, abs=0.1)
    assert summary["ego_speed_median_mps"] == pytest.approx(10.0, abs=0.1)
    assert summary["best_headway_s"] == synthetic_scan.events["headway_s"].min()
    assert summary["left_at_speed"] == 0 and summary["front_left_calibrated"] is False  # it has no camera_calibration
    # The earliest row in time, though the file holds it last.
    assert (summary["time_of_day"], summary["location"], summary["weather"]) == ("Dawn/Dusk", "location_phx", "rain")


def test_scan_segment_keys_its_events_by_segment(synthetic_scan):
    assert list(synthetic_scan.events.columns) == EVENT_COLUMNS
    assert synthetic_scan.events[["split", "context_name"]].values.tolist() == [["training", CONTEXT]]


# Each threshold, from both sides, on one car's tables.

THRESHOLDS = [  # condition, then lane_change arguments just inside and just outside it
    ("clear_m", {"clear": -2.5}, {"clear": -2.49}),
    ("clear_m_from_the_left", {"clear": 2.5}, {"clear": 2.49}),
    ("inside_m", {"inside": -1.0}, {"inside": -1.01}),
    ("max_crossing_s", {"crossing_s": 6.0}, {"crossing_s": 6.1}),
    ("min_stay_s", {"stay_s": 1.0}, {"stay_s": 0.9}),
    ("stay_m", {"settled": -1.5}, {"settled": -1.51}),
    ("max_gap_m", {"front_gap_m": 30.0}, {"front_gap_m": 30.01}),
    ("ahead", {"front_gap_m": 0.01}, {"front_gap_m": np.nan}),  # tracks hold NaN for a box not past the bumper
    ("max_heading_rad", {"heading_rad": -0.5}, {"heading_rad": 0.51}),
    ("along_the_path_before_onset", {"yaw_world_rad": 0.5}, {"yaw_world_rad": -0.51}),  # the ego's path runs along x
    ("min_object_speed_mps", {"speed_mps": 2.0}, {"speed_mps": 1.99}),
    ("object_speed_known", {"speed_mps": 12.0}, {"speed_mps": np.nan}),
    ("min_ego_speed_mps", {"ego_speed": 3.0}, {"ego_speed": 2.99}),
    # Entry is 29 m along the path, which ends at the ego's last pose, 59 m along.
    ("path_start", {"along_path_m": -29.0}, {"along_path_m": -29.01}),
    ("path_end_allowance_m", {"along_path_m": 38.0}, {"along_path_m": 38.01}),
    ("path_known", {"along_path_m": 5.0}, {"along_path_m": np.nan}),
    ("front_half_fov_deg", at_bearing(24.9), at_bearing(25.1)),
    ("front_half_fov_deg_to_the_right", at_bearing(-24.9), at_bearing(-25.1)),
]


@pytest.mark.parametrize("kwargs, found", [
    *[pytest.param(inside, 1, id=f"{name}-inside") for name, inside, _ in THRESHOLDS],
    *[pytest.param(outside, 0, id=f"{name}-outside") for name, _, outside in THRESHOLDS],
])
def test_each_condition_can_fail_on_its_own(kwargs, found):
    assert len(find_cut_ins(*lane_change(**kwargs), CONFIG)) == found


@pytest.mark.parametrize("field, value, kwargs", [
    ("clear_m", 2.49, {"clear": -2.49}),
    ("inside_m", 1.01, {"inside": -1.01}),
    ("max_crossing_s", 6.1, {"crossing_s": 6.1}),
    ("min_stay_s", 0.9, {"stay_s": 0.9}),
    ("stay_m", 1.51, {"settled": -1.51}),
    ("max_gap_m", 30.01, {"front_gap_m": 30.01}),
    ("max_heading_rad", 0.51, {"heading_rad": 0.51}),
    ("max_heading_rad", 0.51, {"yaw_world_rad": 0.51}),
    ("min_object_speed_mps", 1.99, {"speed_mps": 1.99}),
    ("min_ego_speed_mps", 2.99, {"ego_speed": 2.99}),
    ("path_end_allowance_m", 8.01, {"along_path_m": 38.01}),
    ("front_half_fov_deg", 25.2, at_bearing(25.1)),
    ("front_camera_x_m", 0.0, at_bearing(25.1)),  # from the rear axle, the same car sits 23.5 degrees out
])
def test_each_threshold_comes_from_the_config(field, value, kwargs):
    assert len(find_cut_ins(*lane_change(**kwargs), replace(CONFIG, **{field: value}))) == 1


def test_an_entry_past_the_path_end_is_marked():
    """Entry is 29 m along a path that ends at 59 m: 30 m ahead of it is 0.01 m too far for the path itself."""
    assert find_cut_ins(*lane_change(along_path_m=30.0), CONFIG)["past_path_end"].tolist() == [False]
    assert find_cut_ins(*lane_change(along_path_m=30.01), CONFIG)["past_path_end"].tolist() == [True]


def test_crossing_traffic_passes_straight_through_and_is_not_found(tmp_path, stage1):
    """A car crossing 15 m ahead at 10 m/s spends 0.2 s in the path. Only the stay rule turns it away here."""
    t = np.arange(60) / 10.0
    cars = {"CROSSER": (np.full(60, 45.0), 10.0 * (t - 3.05), np.full(60, np.pi / 2.0))}
    ref = write_scene(tmp_path, "crossing", driving(10.0 * t), cars)
    any_heading = replace(CONFIG, max_heading_rad=np.pi)
    assert scan_segment(tmp_path, ref, stage1, any_heading)[0].empty
    assert len(scan_segment(tmp_path, ref, stage1, replace(any_heading, min_stay_s=0.0))[0]) == 1


def test_a_stopped_ego_gives_no_events(tmp_path, stage1):
    """A car pulling in front of a parked ego is no sudden cut-in. The same move ahead of a moving ego is one."""
    t = np.arange(60) / 10.0
    ahead, lateral = 6.0 + 3.0 * t, -3.5 * np.clip((4.0 - t) / 2.0, 0.0, 1.0)  # in from the right, from 2 s to 4 s
    for name, ego_speed, found in (("moving", 10.0, 1), ("stopped", 0.0, 0)):
        ego = driving(ego_speed * t)
        ref = write_scene(tmp_path, name, ego, {"MERGER": (ego[0] + ahead, lateral, np.zeros(60))})
        assert len(scan_segment(tmp_path, ref, stage1, CONFIG)[0]) == found


def test_a_car_turning_in_from_a_side_street_is_no_cut_in(tmp_path, stage1):
    """It comes into the path 25 m ahead of the ego, nearly straight by then, but it never changed lanes. Its gap
    closes as it turns, so found, it would rank above every cut-in whose gap opens. Two seconds before onset it's
    still on the side street, moving, so the heading turns it away, not the speed."""
    scene = (driving(6.0 * TEN_HZ), {"TURNS": turning_in(6.0 * 9.0 + 25.0)})
    assert scanned(tmp_path, {"turn_in": scene}, stage1).empty
    assert len(scanned(tmp_path, {"turn_in": scene}, stage1, replace(CONFIG, max_heading_rad=np.pi))) == 1


def test_a_cut_in_as_the_ego_leaves_a_turn_is_still_found(tmp_path, stage1):
    """The ego turns onto the road from a side street. A car on the road starts over from the next lane 1.5 s
    before the ego leaves the turn, so two seconds before onset the ego's nose is still 1.4 rad off the road."""
    cuts = driving(20.0 + 6.0 * (TEN_HZ - 9.0), LANE * (1.0 - smoothstep(TEN_HZ, 7.5, 3.0)))
    assert len(scanned(tmp_path, {"after_turn": (turning_in(0.0), {"CUTS": cuts})}, stage1)) == 1


def test_a_car_pulling_out_of_a_parking_spot_is_no_cut_in(tmp_path, stage1):
    """a10b's shape: parked 3 m right of the ego's line, 30 m ahead of a 9 m/s ego, it pulls away at 2 m/s² and
    moves over in 3 s. Two seconds before onset it was standing still. The same move by a car already rolling at
    3 m/s in the next lane is a cut-in."""
    def pulls_out(speed):
        return driving(102.0 + speed * (TEN_HZ - 8.0) + np.clip(TEN_HZ - 8.0, 0.0, None) ** 2,
                       -3.0 * (1.0 - smoothstep(TEN_HZ, 8.3, 3.0)))

    events = scanned(tmp_path, {name: (driving(9.0 * TEN_HZ), {"PULLS": pulls_out(speed)})
                                for name, speed in (("parked", 0.0), ("rolling", 3.0))}, stage1)
    assert events["context_name"].tolist() == ["rolling"]
    standing_still_counts = replace(CONFIG, min_object_speed_mps=0.0)
    assert len(scanned(tmp_path, {"parked": (driving(9.0 * TEN_HZ), {"PULLS": pulls_out(0.0)})}, stage1,
                       standing_still_counts)) == 1


@pytest.mark.parametrize("bearings, found", [((40.0, 5.0), 1), ((5.0, 40.0), 0)],
                         ids=["seen_only_at_entry", "seen_only_at_onset"])
def test_conditions_are_judged_at_entry_not_at_onset(bearings, found):
    """A car overtaking from the side is often outside FRONT when it starts across, and that's fine."""
    tracks, ego = lane_change()
    at_onset, at_entry = (at_bearing(degrees)["center_y_m"] for degrees in bearings)
    tracks = tracks.assign(center_x_m=21.5, center_y_m=np.where(tracks["t_s"] < 2.85, at_onset, at_entry))
    assert len(find_cut_ins(tracks, ego, CONFIG)) == found


def before_onset(row: int, config=CONFIG, **columns) -> int:
    """How many events in lane_change with onset at 2.9 s, so 2 s before it is row 9, given `columns` in `row`."""
    tracks, ego = lane_change(clear_s=3.0)
    for name, value in columns.items():
        tracks.loc[row, name] = value
    return len(find_cut_ins(tracks, ego, config))


@pytest.mark.parametrize("columns", [{"yaw_world_rad": 1.2}, {"speed_mps": 1.99}, {"speed_mps": np.nan}],
                         ids=["across_the_road", "standing_still", "speed_unknown"])
def test_the_car_was_driving_along_the_road_own_move_lead_s_before_onset(columns):
    """Onset is at 2.9 s, so it's the car's sample at 0.9 s that counts. A car turning in from a side street is
    across the road then, and one pulling out of a parking spot is standing still."""
    assert [before_onset(row, **columns) for row in (8, 9, 10)] == [1, 0, 1]
    assert before_onset(8, replace(CONFIG, own_move_lead_s=2.1), **columns) == 0


def test_a_car_moving_at_min_object_speed_mps_before_onset_counts():
    assert before_onset(9, speed_mps=2.0) == 1
    assert before_onset(9, replace(CONFIG, min_object_speed_mps=1.99), speed_mps=1.99) == 1


def test_the_heading_before_onset_is_against_the_path_where_the_car_is_not_the_egos_nose():
    """Until 0.9 s the ego is still turning, its nose 0.8 rad left of the road it then drives. The car is on that
    road ahead, along it, so it counts. A car pointing where the ego's nose points is across the road."""
    tracks, ego = lane_change(clear_s=3.0)
    turning = ego["t_s"] < 0.95
    ego = ego.assign(yaw_rad=np.where(turning, 0.8, 0.0))
    assert len(find_cut_ins(tracks.assign(heading_rad=np.where(turning, -0.75, 0.05)), ego, CONFIG)) == 1
    assert find_cut_ins(tracks.assign(yaw_world_rad=np.where(turning, 0.8, 0.05)), ego, CONFIG).empty


def test_a_path_heading_west_still_runs_west_between_two_poses():
    """Heading west, the ego's yaw flips between pi and -pi from pose to pose. The car sits halfway between two,
    heading west too, as -pi."""
    tracks, ego = lane_change(along_path_m=5.5, yaw_world_rad=-np.pi)
    ego = ego.assign(yaw_rad=np.where(np.arange(len(ego)) % 2, -3.13, 3.13))
    assert len(find_cut_ins(tracks, ego, CONFIG)) == 1


def test_a_car_that_waits_in_the_next_lane_is_timed_from_when_it_moves():
    """Timed from its first clear sample, this 1 s crossing would read 15 s and fail the 6 s limit."""
    events = find_cut_ins(*tables([-3.5] * 150 + [-1.75] * 9 + [0.0] * 20), CONFIG)
    assert events["crossing_s"].tolist() == pytest.approx([1.0])


def test_each_entry_is_its_own_event():
    lateral = [-3.0] * 20 + [-1.75] * 4 + [0.0] * 15 + [3.0] * 10 + [1.75] * 4 + [0.0] * 15
    events = find_cut_ins(*tables(lateral), CONFIG)
    assert events["entry_t_s"].tolist() == pytest.approx([2.4, 5.3])


def test_wandering_inside_the_path_is_one_entry():
    """Back toward the line and in again, without clearing the path first, is the same cut-in."""
    lateral = [-3.0] * 20 + [-1.75] * 4 + [0.0] * 12 + [-1.5] * 3 + [0.0] * 12
    assert len(find_cut_ins(*tables(lateral), CONFIG)) == 1


def test_a_car_already_in_the_path_must_clear_it_before_it_can_cut_in():
    assert find_cut_ins(*tables([0.0] * 20 + [-1.75] * 5 + [0.0] * 20), CONFIG).empty


def test_a_track_that_ends_in_the_path_has_stayed_as_long_as_it_could():
    assert len(find_cut_ins(*lane_change(stay_s=0.5, tail_s=0.0), CONFIG)) == 1


@pytest.mark.parametrize("unseen_as", ["missing", "unknown"])
def test_a_hole_after_entry_is_no_track_end(unseen_as):
    """A car weaving through the path, unseen as a vehicle from 0.2 s after entry until it's 3 m out the other side.

    It was in the path before the hole, but it never stayed: the first sample past the stay must be in it too.
    """
    tracks, ego = tables([3.0] * 20 + [1.75] * 4 + [0.5, 0.0] + [-1.75] * 4 + [-3.0] * 20)  # entry at 2.4 s
    unseen = tracks["t_s"].between(2.55, 3.75)
    tracks = tracks[~unseen] if unseen_as == "missing" else tracks.assign(type=tracks["type"].mask(unseen, "unknown"))
    assert find_cut_ins(tracks, ego, CONFIG).empty


def test_a_cut_in_unseen_for_a_while_after_entry_is_still_found():
    tracks, ego = lane_change(stay_s=5.0)  # entry at 2.9 s
    assert len(find_cut_ins(tracks[~tracks["t_s"].between(2.95, 4.45)], ego, CONFIG)) == 1


def test_label_noise_at_the_edge_of_the_path_does_not_lose_a_cut_in():
    """Noise takes a label under inside_m a sample early, then back over it: still in the path for the stay."""
    events = find_cut_ins(*tables([-3.0] * 20 + [-2.0] * 5 + [-0.98, -1.02, -0.9] + [-0.6] * 15), CONFIG)
    assert events["entry_t_s"].tolist() == pytest.approx([2.5])


@pytest.mark.parametrize("box_type", ["pedestrian", "cyclist", "sign", "unknown"])
def test_only_vehicles_cut_in(box_type):
    assert find_cut_ins(*lane_change(type=box_type), CONFIG).empty


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["from_the_right", "from_the_left"])
def test_severity_is_read_at_entry_and_over_the_crossing(side):
    # Onset is the last 3.0 out at 1.9 s, entry the 0.9 at 2.3 s. The jumps before onset and after entry don't count.
    lateral = side * np.array([-6.0] * 10 + [-3.0] * 10 + [-2.4, -1.9, -1.6, -0.9] + [0.1] * 20)
    tracks, ego = tables(lateral)
    steps = np.arange(len(lateral))
    # Each changes every sample, so a value read from the wrong one shows.
    tracks = tracks.assign(front_gap_m=10.0 + 0.1 * steps, v_rel_ego_x_mps=-0.1 * steps)
    ego = ego.assign(speed_mps=5.0 + 0.1 * steps)
    [event] = find_cut_ins(tracks, ego, CONFIG).itertuples()
    assert (event.onset_t_s, event.entry_t_s, event.crossing_s) == pytest.approx((1.9, 2.3, 0.4))
    assert event.headway_s == pytest.approx(12.3 / 7.3)
    assert event.closing_mps == pytest.approx(2.3)
    assert event.required_decel_mps2 == pytest.approx(4.3**2 / (2.0 * 14.3))  # the track's last sample, closing fastest
    assert event.severity_mps2 == event.required_decel_mps2  # more than the ego's 0.5 m/s² of braking
    assert event.peak_lateral_mps == pytest.approx(7.0)  # the -1.6 to -0.9 step, 0.7 m in 0.1 s
    assert event.own_move_m == pytest.approx(3.0 - 0.9)
    assert (event.onset_timestamp_micros, event.entry_timestamp_micros) == (T0 + 1_900_000, T0 + 2_300_000)
    assert event.alias == "V001" and event.laser_object_id == "CAR"


@pytest.mark.parametrize("rows, values, beside, braking", [
    ([18, 19], [-9.0, -3.0], False, -3.0),  # just before onset, and at it: a planner brakes as the car starts over
    ([18, 28, 59], [-9.0, -2.5, -2.0], False, -2.5),  # before onset, before entry, and 3.0 s after it
    ([59, 60], [-2.0, -8.0], False, -2.0),  # 3.0 s after entry, and 3.1 s after
    ([24], [-4.0], True, -4.0),  # while the car is still beside the ego, with no gap yet: the span is the ego's
])
def test_braking_is_the_egos_lowest_a_long_from_onset_to_three_seconds_after_entry(rows, values, beside, braking):
    tracks, ego = lane_change(stay_s=5.0)  # onset at 1.9 s, row 19, and entry at 2.9 s, row 29
    if beside:
        tracks.loc[19:28, "front_gap_m"] = np.nan  # not past the bumper until entry
    a_long = np.zeros(len(ego))
    a_long[rows] = values
    [event] = find_cut_ins(tracks, ego.assign(a_long_mps2=a_long), CONFIG).itertuples()
    assert (event.ego_min_a_long_mps2, event.severity_mps2) == (braking, -braking)  # more than the 0.17 m/s² needed


def test_the_braking_window_comes_from_the_config():
    tracks, ego = lane_change(stay_s=5.0)
    a_long = np.zeros(len(ego))
    a_long[[59, 60]] = [-2.0, -8.0]
    ego = ego.assign(a_long_mps2=a_long)
    assert find_cut_ins(tracks, ego, replace(CONFIG, braking_window_s=3.1))["ego_min_a_long_mps2"].item() == -8.0


def test_events_come_in_entry_order_across_tracks():
    """V002 enters first though its ID sorts last, which is the order tracks are searched in."""
    first, ego = lane_change(crossing_s=3.0)
    second, _ = lane_change(crossing_s=1.0)
    tracks = pd.concat([first, second.assign(laser_object_id="ZZZ", alias="V002")]).sort_values(TIME)
    assert find_cut_ins(tracks, ego, CONFIG)["alias"].tolist() == ["V002", "V001"]


def test_no_events_still_have_their_columns_and_types():
    """Or a quiet segment would turn events.parquet's timestamps into objects when segments are concatenated."""
    events = find_cut_ins(*lane_change(type="pedestrian"), CONFIG)
    assert list(events.columns) == EVENT_COLUMNS[2:]
    assert events["entry_timestamp_micros"].dtype == np.int64 and events["headway_s"].dtype == np.float64
    assert events["past_path_end"].dtype == bool


def test_find_cut_ins_leaves_its_tables_alone():
    tracks, ego = lane_change()
    before = tracks.copy(), ego.copy()
    find_cut_ins(tracks, ego, CONFIG)
    pd.testing.assert_frame_equal(tracks, before[0])
    pd.testing.assert_frame_equal(ego, before[1])


# The braking needed, from onset.

def closing_once(row: int, **columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """lane_change with a 5 s stay, onset in row 19 and entry in row 29, and the gap opening everywhere but in `row`,
    given `columns`. The ego never brakes, so its severity is the braking it needed."""
    tracks, ego = lane_change(stay_s=5.0, v_rel_ego_x_mps=1.0)
    for name, value in columns.items():
        tracks.loc[row, name] = value
    return tracks, ego.assign(a_long_mps2=0.0)


CLOSING = {"front_gap_m": 3.0, "v_rel_ego_x_mps": -2.0}
NEEDED = 2.0**2 / (2.0 * 3.0)  # closing at 2 m/s, 3 m short of the bumper


@pytest.mark.parametrize("row, columns, needed, config", [
    (40, {}, 0.0, CONFIG),  # the gap never closes: 0, not NaN
    (40, CLOSING, NEEDED, CONFIG),
    (40, {"front_gap_m": 3.0, "v_rel_ego_x_mps": -1e-6}, 1e-12 / 6.0, CONFIG),  # closing, however slowly
    (40, {"front_gap_m": 3.0, "v_rel_ego_x_mps": 0.0}, 0.0, CONFIG),
    (40, {"front_gap_m": 0.0, "v_rel_ego_x_mps": -2.0}, 0.0, CONFIG),
    (40, {"front_gap_m": np.nan, "v_rel_ego_x_mps": -2.0}, 0.0, CONFIG),  # not past the bumper
    (40, {**CLOSING, "lateral_offset_m": -1.5}, NEEDED, CONFIG),
    (40, {**CLOSING, "lateral_offset_m": 1.51}, 0.0, CONFIG),  # out of the path after entry
    (40, {**CLOSING, "lateral_offset_m": -1.51}, 0.0, CONFIG),  # out of it on the other side
    (40, {**CLOSING, "lateral_offset_m": 1.51}, NEEDED, replace(CONFIG, stay_m=1.51)),
    (29, CLOSING, NEEDED, CONFIG),  # at entry
    (28, CLOSING, NEEDED, CONFIG),  # before entry, 1.75 m out: still crossing, so no path test
    (19, CLOSING, NEEDED, CONFIG),  # at onset, 3 m out
    (18, CLOSING, 0.0, CONFIG),  # before onset
    (59, CLOSING, NEEDED, CONFIG),  # 3.0 s after entry
    (60, CLOSING, 0.0, CONFIG),  # 3.1 s after it
    (60, CLOSING, NEEDED, replace(CONFIG, braking_window_s=3.1)),
])
def test_the_braking_needed_counts_samples_from_onset_ahead_and_closing_and_in_the_path_from_entry(row, columns,
                                                                                                   needed, config):
    [event] = find_cut_ins(*closing_once(row, **columns), config).itertuples()
    assert (event.required_decel_mps2, event.severity_mps2) == (pytest.approx(needed), pytest.approx(needed))


def test_the_braking_needed_is_its_largest_over_the_samples():
    tracks, ego = closing_once(40, **CLOSING)
    tracks.loc[45, ["front_gap_m", "v_rel_ego_x_mps"]] = 2.0, -3.0
    assert find_cut_ins(tracks, ego, CONFIG)["required_decel_mps2"].item() == pytest.approx(9.0 / 4.0)


@pytest.mark.parametrize("gap, needed", [
    (0.0201, 4.0 / 0.0402),  # just under the cap, left alone
    (0.02, 100.0),
    (0.0199, 100.0),
    (1e-300, 100.0),  # 2e300 uncapped: the report refuses anything from 1e12
])
def test_the_braking_needed_is_capped_far_past_what_any_car_can_do(gap, needed):
    """A gap of a millimeter or less is label noise as much as a gap, and its need could otherwise be any size."""
    [event] = find_cut_ins(*closing_once(40, front_gap_m=gap, v_rel_ego_x_mps=-2.0), CONFIG).itertuples()
    assert event.required_decel_mps2 == pytest.approx(needed)


@pytest.mark.parametrize("a_long, severity", [
    (-0.1, NEEDED),  # it needed more than it braked
    (-4.5, 4.5),  # it braked more than it needed
    (np.nan, NEEDED),  # unknown braking counts as none
    (0.5, NEEDED),  # speeding up isn't braking
])
def test_severity_is_the_larger_of_the_braking_needed_and_the_braking_done(a_long, severity):
    tracks, ego = closing_once(40, **CLOSING)
    [event] = find_cut_ins(tracks, ego.assign(a_long_mps2=a_long), CONFIG).itertuples()
    assert event.severity_mps2 == pytest.approx(severity)


@pytest.mark.parametrize("a_long", [0.5, np.nan])
def test_severity_is_never_below_0(a_long):
    """The gap opens throughout, and the ego speeds up or its braking is unknown."""
    tracks, ego = closing_once(40)
    [event] = find_cut_ins(tracks, ego.assign(a_long_mps2=a_long), CONFIG).itertuples()
    assert (event.required_decel_mps2, event.severity_mps2) == (0.0, 0.0)


# The car's own move, against the ego's line from before onset.

def test_the_own_move_is_against_the_egos_line_own_move_lead_s_before_onset():
    """Onset is at 2.9 s. Between 0.9 s and 1.0 s the ego moves 3 m into the car's lane: from 0.9 s's pose, the
    car came 3 m nearer, and from 1.0 s's, it went 3 m away."""
    tracks, ego = lane_change(clear_s=3.0)
    ego = ego.assign(pos_local_y_m=np.where(ego["t_s"] < 0.95, 0.0, -3.0))
    assert find_cut_ins(tracks, ego, CONFIG)["own_move_m"].tolist() == pytest.approx([3.0])
    later = replace(CONFIG, own_move_lead_s=1.9)
    assert find_cut_ins(tracks, ego, later)["own_move_m"].tolist() == pytest.approx([-3.0])


def test_scan_toml_leads_by_2_s():
    """At 1 s, the pose already sat inside a slow ego lane change often enough to pass it off as the car's move."""
    assert load_scan_config(SCAN_CONFIG_PATH).own_move_lead_s == 2.0


def test_a_lead_from_before_the_segment_takes_its_first_pose():
    tracks, ego = tables([-3.0] * 4 + [-1.75] * 5 + [0.0] * 20)  # onset at 0.3 s
    ego = ego.assign(pos_local_y_m=np.where(ego["t_s"] < 0.05, 0.0, -3.0))
    assert find_cut_ins(tracks, ego, CONFIG)["own_move_m"].tolist() == pytest.approx([3.0])


def on_a_bend(speed: float, yaw_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """lane_change's car on a 100 m radius bend to the left, as seen from the ego's pose 2 s before onset, 9 m along x.

    At onset it's 3 m right of the ego's lane 20 m around the bend, and at entry in it, 30 m around. The pose has
    this speed and yaw rate, and from there the ego keeps driving the circle they give, at lane_change's 10 m/s.
    """
    tracks, ego = lane_change(clear_s=3.0)  # onset in row 29, entry in row 39
    ego.loc[9, ["speed_mps", "yaw_rate_radps"]] = speed, yaw_rate
    ego.loc[10:, "yaw_rate_radps"] = 10.0 * scan._curvature(ego.loc[[9]])[0]
    for row, (along, left) in ((29, (20.0, -3.0)), (39, (30.0, 0.0))):
        x, y = on_road(along, left, 0.01)
        tracks.loc[row, ["pos_local_x_m", "pos_local_y_m"]] = 9.0 + x, y
    return tracks, ego


@pytest.mark.parametrize("speed, yaw_rate, own_move", [
    (10.0, 0.1, 3.0),        # the circle the ego was driving
    (10.0, 0.0, -3.52),      # a straight line misjudges it
    (1.0, 0.01, 3.0),        # slow, but turning
    (0.99, 0.0099, -3.52),   # too slow to tell a turn from noise, so a straight line
])
def test_the_own_move_is_against_the_circle_the_ego_was_driving(speed, yaw_rate, own_move):
    assert find_cut_ins(*on_a_bend(speed, yaw_rate), CONFIG)["own_move_m"].item() == pytest.approx(own_move, abs=0.01)


def test_the_circle_bends_no_tighter_than_a_car_can_turn():
    """About 5 m, a curvature of 0.2: a noisy yaw rate can't bend it further."""
    def own_move(yaw_rate):
        return find_cut_ins(*on_a_bend(10.0, yaw_rate), CONFIG)["own_move_m"].item()

    assert own_move(50.0) == own_move(2.0) != own_move(1.9)
    assert own_move(-50.0) == own_move(-2.0) != own_move(-1.9)


def steered(rows: dict, config=CONFIG, ego_speed=10.0) -> tuple[float, float]:
    """lane_change's event's lateral_change_mps2 and own_move_m, with the ego's (yaw rate, speed) in these rows.

    Onset is at 2.9 s, row 29, so the pose is at 0.9 s, row 9, the window ends at 2.4 s, row 24, and entry is in
    row 39. On the pose's straight line a yaw rate pulls the ego sideways by itself times the speed: 0.05 rad/s at
    10 m/s is 0.5 m/s².
    """
    tracks, ego = lane_change(clear_s=3.0, ego_speed=ego_speed)
    for row, values in rows.items():
        ego.loc[row, ["yaw_rate_radps", "speed_mps"]] = values
    [event] = find_cut_ins(tracks, ego, config).itertuples()
    return event.lateral_change_mps2, event.own_move_m


@pytest.mark.parametrize("rows, config, change", [
    ({24: (0.05, 10.0)}, CONFIG, 0.5),  # at the window's end, half a second before onset
    ({24: (0.0501, 10.0)}, CONFIG, 0.501),
    ({24: (-0.0501, 10.0)}, CONFIG, 0.501),  # the other way
    ({25: (1.0, 10.0)}, CONFIG, 0.0),  # after it, when the ego may react to the car's first drift
    ({25: (1.0, 10.0)}, replace(CONFIG, steer_window_end_s=0.4), 10.0),
    ({29: (1.0, 10.0)}, CONFIG, 0.0),  # at onset
    ({39: (1.0, 10.0)}, CONFIG, 0.0),  # at entry
    ({8: (1.0, 10.0)}, CONFIG, 0.0),  # before the pose
    ({10: (0.0501, 10.0)}, CONFIG, 0.501),  # just after it
    ({9: (0.0501, 10.0)}, CONFIG, 0.501),  # the pose's own, so every row after it is off its circle
    ({20: (0.05, 10.02)}, CONFIG, 0.501),  # the same yaw rate, a little faster: each row's own speed
    ({20: (0.05, 5.0), 24: (0.0, 12.0)}, CONFIG, 0.25),  # not the window's fastest
    ({9: (0.03, 10.0), 20: (0.06, 10.0)}, CONFIG, 0.3),  # off the pose's circle, not a straight line's
    ({20: (0.03, 10.0), 24: (-0.03, 10.0)}, CONFIG, 0.3),  # nor the window end's
    ({20: (np.nan, 10.0)}, CONFIG, np.nan),  # unknown, so nothing says the ego kept to the circle
    ({20: (0.0, np.nan)}, CONFIG, np.nan),
    ({9: (0.0, np.nan)}, CONFIG, np.nan),  # the pose's own speed too
    ({20: (np.nan, 0.99)}, CONFIG, 0.0),  # too slow to tell a turn from noise, so on the pose's straight line
])
def test_the_pull_off_the_poses_circle_counts_from_the_pose_to_steer_window_end_s_before_onset(rows, config, change):
    """The circle stands for the lane only while the ego keeps driving it. Leaving it pulls the ego sideways by its
    curvature's change times its speed squared. The own move is measured either way: _rank weighs the pull."""
    lateral_change, own_move = steered(rows, config)
    assert lateral_change == pytest.approx(change, nan_ok=True)
    assert np.isfinite(own_move)


def test_the_window_always_holds_the_poses_own_frame():
    """Onset at 0.3 s: the pose is the segment's first frame, after the window's end would be, so the window is that
    frame alone. Every later frame turns hard, and none counts. The pose's own pull is 0, or unknown with its speed."""
    tracks, ego = tables([-3.0] * 4 + [-1.75] * 5 + [0.0] * 20)
    ego.loc[1:, "yaw_rate_radps"] = 1.0
    assert find_cut_ins(tracks, ego, CONFIG)["lateral_change_mps2"].tolist() == [0.0]
    ego.loc[0, "speed_mps"] = np.nan
    assert np.isnan(find_cut_ins(tracks, ego, CONFIG)["lateral_change_mps2"].item())


def test_a_pose_after_lost_frames_still_has_its_own_frame_in_the_window():
    """Frames are lost from 1.0 s to 4.4 s, labels and poses alike. Onset is at 4.9 s, so the frame nearest 2 s
    before it is 4.5 s, past where the window would end. The window is that frame alone: a turn at 4.7 s isn't in it."""
    tracks, ego = lane_change(clear_s=5.0)
    ego.loc[47, "yaw_rate_radps"] = 1.0
    kept = ~tracks["t_s"].between(0.95, 4.45).to_numpy()
    [event] = find_cut_ins(tracks[kept], ego[kept], CONFIG).itertuples()
    assert (event.onset_t_s, event.lateral_change_mps2) == (pytest.approx(4.9), 0.0)


def test_a_frame_1_ms_past_the_windows_end_is_out():
    """Real frames come 99.9 to 100.1 ms apart, so the frame 0.5 s before onset often lands a hair past the window's
    end. 1 ms past, it's out, and its turn doesn't count: on real data the window ends 0.5 to 0.6 s before onset."""
    tracks, ego = lane_change(clear_s=3.0)
    ego.loc[24, "yaw_rate_radps"] = 0.0501  # 0.501 m/s² at 10 m/s, were it in
    for table in (tracks, ego):
        table.loc[24, TIME] += 1_000  # 2.401 s, with onset at 2.9 s
    assert find_cut_ins(tracks, ego, CONFIG)["lateral_change_mps2"].item() == 0.0


def test_scan_toml_ends_the_window_half_a_second_before_onset():
    assert load_scan_config(SCAN_CONFIG_PATH).steer_window_end_s == 0.5


@pytest.mark.parametrize("ego_speed, yaw_rate, pull", [(3.0, 0.06, 0.18), (5.0, 0.1, 0.5), (10.0, 0.2, 2.0)])
def test_a_curvature_change_pulls_harder_the_faster_the_ego(ego_speed, yaw_rate, pull):
    """The same change of 0.02 per meter pulls 0.18 m/s² at 3 m/s and 0.5 at 5 m/s, within the limit, and 2 m/s² at
    10 m/s, a steer."""
    assert steered({20: (yaw_rate, ego_speed)}, ego_speed=ego_speed)[0] == pytest.approx(pull)


def test_braking_on_a_bend_is_no_steer():
    """From the pose to onset the ego slows from 14 m/s to 6 on a bend of 100 m radius. Its curvature holds, so it keeps
    to its circle, though its sideways acceleration falls from 2 m/s² to 0.4."""
    speeds = np.linspace(14.0, 6.0, 21)  # rows 9 to 29
    lateral_change, own_move = steered({row: (0.01 * speed, speed) for row, speed in zip(range(9, 30), speeds)})
    assert lateral_change == pytest.approx(0.0, abs=1e-9) and np.isfinite(own_move)


def test_scan_toml_takes_a_steer_at_half_a_meter_per_second_squared():
    assert load_scan_config(SCAN_CONFIG_PATH).max_lateral_change_mps2 == 0.5


CUT_IN = cuts_in(8.0, 3.0)  # in from the right lane over 3 s from 8 s


@pytest.mark.parametrize("curvature", [0.0, 0.01, -0.01], ids=["straight", "left_bend", "right_bend"])
def test_a_real_cut_in_keeps_an_own_move_over_1_m(tmp_path, stage1, curvature):
    """A car 2 m/s faster than the ego cuts in, on a straight road and on bends of 100 m radius. Its own move is
    how much nearer the ego's lane it came, from onset to entry."""
    scene = (driving(12.0 * TEN_HZ, curvature=curvature), {"CUTS": driving(16.0 + 14.0 * TEN_HZ, CUT_IN, curvature)})
    [event] = scanned(tmp_path, {"cut_in": scene}, stage1).itertuples()
    offset = np.abs(np.interp([event.onset_t_s, event.entry_t_s], TEN_HZ, CUT_IN))
    assert event.own_move_m == pytest.approx(offset[0] - offset[1], abs=0.05)
    assert event.own_move_m > 1.0


@pytest.mark.parametrize("amplitude, speeds", [(0.03, (6.0, 10.0, 15.0)), (0.05, (10.0, 15.0))])
def test_lane_keeping_wander_is_no_steer_at_any_speed(tmp_path, stage1, amplitude, speeds):
    """The ego weaves this far either side of its lane's centre every 4 s, at four phases, while a car 2 m/s slower
    cuts in from the right over 2.5 s, 8 m past the bumper, and the ego eases off at 1 m/s² from entry. The weave
    pulls the ego sideways by up to amplitude (2 pi / 4 s)² either way, at any speed. Onset is at 8.9 s, so the
    window runs from the pose at 6.9 s to 8.4 s. Phase 0's pull, the largest, peaks at the window's end, at
    amplitude (2 pi / 4 s)² times how far its sine has swung since the pose."""
    scenes, phases = {}, {}
    for speed in speeds:
        ego = braking(speed, 9.65, 1.0, speed - 2.0)
        car = driving(np.interp(9.65, TEN_HZ, ego) + 14.3 + (speed - 2.0) * (TEN_HZ - 9.65), cuts_in(8.0, 2.5))
        for phase in range(4):
            name = f"{speed:g}_{phase}"
            scenes[name] = (driving(ego, amplitude * np.sin(np.pi * (TEN_HZ + phase) / 2.0)), {"CUTS": car})
            phases[name] = phase
    events = scanned(tmp_path, scenes, stage1)
    assert len(events) == 4 * len(speeds) and (events["own_move_m"] > 1.0).all()
    assert (events["lateral_change_mps2"] < CONFIG.max_lateral_change_mps2).all()
    pulls = events.groupby(events["context_name"].map(phases))["lateral_change_mps2"]
    assert (pulls.max() - pulls.min() < 0.01).all()  # the same at every speed
    pose_s, end_s = 6.9, 8.4  # the window, from 2 s to 0.5 s before onset
    swing = np.sin(np.pi * end_s / 2.0) - np.sin(np.pi * pose_s / 2.0)  # phase 0's weave is sin(pi t / 2)
    assert pulls.max().max() == pytest.approx(swing * amplitude * (np.pi / 2.0) ** 2, rel=0.05)


def nudging(shift: float, start: float, duration: float) -> tuple:
    """A 6 m/s car cuts in 6 m past the bumper of a 10 m/s ego, which brakes at 3 m/s² from half a second into the
    car's move, and moves `shift` away from it from `start`. The car is 2.5 m out at 8.95 s."""
    ego = braking(10.0, 8.5, 3.0, 4.0)
    car = driving(np.interp(9.65, TEN_HZ, ego) + 12.3 + 6.0 * (TEN_HZ - 9.65), cuts_in(8.0, 2.5))
    return driving(ego, shift * smoothstep(TEN_HZ, start, duration)), {"CUTS": car}


@pytest.mark.parametrize("shift, duration", [(0.1, 1.5), (0.2, 1.5), (0.4, 1.5), (0.4, 2.0)])
def test_a_nudge_away_from_the_car_while_braking_is_no_steer(tmp_path, stage1, shift, duration):
    """Half a second after the car is 2.5 m out, the ego moves this far away from it. That's after onset."""
    [event] = scanned(tmp_path, {"nudges": nudging(shift, 9.5, duration)}, stage1).itertuples()
    assert event.own_move_m > 1.0 and event.lateral_change_mps2 == 0.0


@pytest.mark.parametrize("start, config, pull", [
    (8.7, CONFIG, 0.13),
    (8.7, replace(CONFIG, steer_window_end_s=0.0), 0.84),  # a window to onset took it for a steer
    (8.45, CONFIG, 0.86),
])
def test_a_nudge_at_the_cars_first_drift_counts_only_from_before_the_windows_end(tmp_path, stage1, start, config,
                                                                                  pull):
    """The car starts over at 8.0 s, and a planner nudges 0.4 m away from it over 1.5 s. The nudge widens the car's
    offset from the ego's path, so the scan's onset is 9.1 s and the window ends at 8.6 s. A nudge from 8.7 s, 0.4 s
    before onset, pulls only after that: the cut-in ranks on its braking. One from 8.45 s, 0.65 s before onset, pulls
    hardest inside it: the known miss."""
    [event] = scanned(tmp_path, {"nudges": nudging(0.4, start, 1.5)}, stage1, config).itertuples()
    assert (event.onset_t_s, event.lateral_change_mps2) == pytest.approx((9.1, pull), abs=0.01)
    assert event.own_move_m == pytest.approx(1.64, abs=0.01)  # the car's, steer or not


# The ranking.

RANKED = pd.DataFrame([
    ("ego_moved", 0.99, 0.0, 9.0, 0.1),
    ("unknown_move", np.nan, 0.0, 9.0, 0.1),
    ("moved_toward", -3.0, 0.0, 9.5, 0.1),
    ("pulled", 1.6, 0.501, 9.7, 0.1),
    ("unknown_pull", 1.6, np.nan, 9.8, 0.1),
    ("mild", 1.6, 0.0, 0.03, 2.0),
    ("two_events", 1.6, 0.0, 0.5, 1.0),
    ("two_events", 1.6, 0.0, 4.0, 1.0),
    ("hard", 1.0, 0.5, 4.5, 2.0),  # on both lines
    ("braked_far", 1.6, 0.0, 1.0, 1.5),
    ("braked_near", 1.6, 0.0, 1.0, 0.5),
    ("tied_b", 1.6, 0.0, 0.0, 1.0),
    ("tied_a", 1.6, 0.0, 0.0, 1.0),
], columns=["context_name", "own_move_m", "lateral_change_mps2", "severity_mps2", "headway_s"]).assign(split="training")


def test_segments_rank_by_own_move_and_pull_then_severity_then_headway_then_name():
    """An own move far below 0 is the ego moving toward the car, so it goes last too, and so does an unknown one. So
    does a pull past max_lateral_change_mps2 before onset, or an unknown one, whatever the own move: the ego may have
    steered. A segment ranks by its most severe event."""
    assert scan._rank(RANKED, CONFIG)["context_name"].tolist() == [
        "hard", "two_events", "braked_near", "braked_far", "mild", "tied_a", "tied_b", "unknown_pull", "pulled",
        "moved_toward", "ego_moved", "unknown_move"]


@pytest.mark.parametrize("field, value, first", [("min_own_move_m", 0.98, "ego_moved"),
                                                 ("max_lateral_change_mps2", 0.501, "pulled")])
def test_both_lines_come_from_the_config(field, value, first):
    assert scan._rank(RANKED, replace(CONFIG, **{field: value}))["context_name"].iloc[0] == first


def test_a_split_breaks_a_tie_before_a_name_does():
    events = pd.DataFrame({"split": ["validation", "training"], "context_name": ["a", "b"], "own_move_m": 1.6,
                           "lateral_change_mps2": 0.0, "severity_mps2": 1.0, "headway_s": 1.0})
    assert scan._rank(events, CONFIG)["split"].tolist() == ["training", "validation"]


def test_the_egos_own_lane_change_ranks_below_a_real_cut_in(tmp_path, stage1):
    """The ego slides in behind a slower car that holds its lane, so that gap closes, while a real cut-in's gap
    opens. By severity alone, the ego's own move would rank first."""
    events = scanned(tmp_path, {
        "car_moved": (driving(12.0 * TEN_HZ), {"CUTS": driving(16.0 + 14.0 * TEN_HZ, CUT_IN)}),
        "ego_moved": (driving(11.0 * TEN_HZ, LANE * smoothstep(TEN_HZ, 8.0, 3.0)),
                      {"HOLDS": driving(38.0 + 10.0 * TEN_HZ, LANE)}),
    }, stage1).set_index("context_name")
    assert events.loc["ego_moved", "own_move_m"] < 1.0 <= events.loc["car_moved", "own_move_m"]
    assert events.loc["car_moved", "severity_mps2"] == pytest.approx(0.0, abs=1e-9)  # nothing closes, nobody brakes
    assert events.loc["ego_moved", "severity_mps2"] > 0.01
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["car_moved", "ego_moved"]


def test_an_egos_slow_lane_change_ranks_below_a_real_cut_in(tmp_path, stage1):
    """a1's shape: the ego takes 8 s to move over behind a car 1.5 m/s slower that holds its lane, braking gently to
    match it. A second before onset the ego was already part way over, so from there the car seemed to move. Two
    seconds before, the ego hadn't started."""
    scenes = {
        "car_moved": (driving(12.0 * TEN_HZ), {"CUTS": driving(16.0 + 14.0 * TEN_HZ, CUT_IN)}),
        "ego_moved": (driving(braking(14.0, 8.0, 0.5, 12.5), LANE * smoothstep(TEN_HZ, 8.0, 8.0)),
                      {"HOLDS": driving(32.0 + 12.5 * TEN_HZ, LANE)}),
    }
    events = scanned(tmp_path, scenes, stage1).set_index("context_name")
    assert events.loc["ego_moved", "severity_mps2"] > events.loc["car_moved", "severity_mps2"]
    assert events.loc["ego_moved", "own_move_m"] < 1.0 <= events.loc["car_moved", "own_move_m"]
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["car_moved", "ego_moved"]
    lead_1_s = scanned(tmp_path, scenes, stage1, replace(CONFIG, own_move_lead_s=1.0))
    assert scan._rank(lead_1_s, CONFIG)["context_name"].tolist() == ["ego_moved", "car_moved"]


def aborted_pass() -> tuple:
    """The benchmark's seed 4 k 3, rounded: the ego pulls out to pass a car 1 m/s slower, gives up and comes back
    behind it, braking at 3 m/s² as that car brakes too."""
    out_and_back = LANE * (smoothstep(TEN_HZ, 6.0, 3.9) - smoothstep(TEN_HZ, 10.4, 5.7))
    return (driving(braking(14.7, 10.4, 3.0, 8.7), out_and_back),
            {"LEAD": driving(27.0 + braking(13.7, 10.4, 2.7, 9.6), -0.2)})


def eased_off_for() -> tuple:
    """A real cut-in: a car 2 m/s slower moves in 10 m past the bumper, and the ego eases off at 1.5 m/s²."""
    eases = braking(12.0, 10.5, 1.5, 9.5)
    return driving(eases), {"CUTS": driving(np.interp(10.0, TEN_HZ, eases) + 16.3 + 10.0 * (TEN_HZ - 10.0), CUT_IN)}


def test_the_egos_own_moves_that_steer_before_onset_rank_below_a_cut_in_it_eased_off_for(tmp_path, stage1):
    """Two non-events whose braking would rank them above a real cut-in, as in the benchmark and its review. No car
    moves sideways, but from the pose 2 s before onset each car seemed to come over by 2 m or more.

    The aborted pass's pose sits late in its pull-out, already turning back, so its circle bends across the lane the
    car holds. In dual turn lanes, the ego leaves a 90° right turn of 15 m radius in the outer lane at 8 m/s and moves
    over to the inner one, over 4 s, behind a car 12 m past its bumper that keeps that lane, and both brake at 2.1 m/s²
    for a light. Its pose comes as it starts the turn. Either way, the ego steered between its pose and the window's
    end, so the own move is doubted. Without the steering test, it would be believed.
    """
    light = braking(8.0, 10.0, 2.1, 1.0)
    u = np.pi * 15.0 / 2.0 + light - np.interp(9.0, TEN_HZ, light)  # it leaves the turn at 9 s
    scenes = {"aborted_pass": aborted_pass(), "cut_in": eased_off_for(),
              "dual_turn": (turning_onto(u, -LANE * smoothstep(TEN_HZ, 9.0, 4.0), 15.0),
                            {"HOLDS": turning_onto(u + 18.3, -LANE, 15.0)})}
    events = scanned(tmp_path, scenes, stage1)
    by_name = events.set_index("context_name")
    assert by_name["severity_mps2"].round(1).to_dict() == {"aborted_pass": 3.0, "cut_in": 1.5, "dual_turn": 2.2}
    assert by_name["lateral_change_mps2"].round(1).to_dict() == {"aborted_pass": 0.9, "cut_in": 0.0, "dual_turn": 1.3}
    assert (by_name.loc[["aborted_pass", "dual_turn"], "own_move_m"] > 2.0).all()
    assert scan._rank(events, CONFIG)["context_name"].tolist() == ["cut_in", "aborted_pass", "dual_turn"]
    no_test = replace(CONFIG, max_lateral_change_mps2=np.inf)
    assert scan._rank(events, no_test)["context_name"].tolist() == ["aborted_pass", "dual_turn", "cut_in"]


def test_a_cut_in_the_ego_brakes_and_nudges_away_for_outranks_an_aborted_pass(tmp_path, stage1):
    """The hard-braking cut-in, an 8 m/s car pulling in front of a 17 m/s ego that needed 3.1 m/s² and braked 4.5,
    with the ego also nudging 0.4 m away from the car over 1.5 s as it brakes. Before entry the nudge pulls it
    sideways by more than 0.5 m/s², but that's after onset, so it isn't counted: the cut-in ranks first, above the
    cut-in the ego eased off for and the aborted pass, whose pull-out came before its onset."""
    scenes = {"aborted_pass": aborted_pass(), "cut_in": eased_off_for(),
              "brakes_and_nudges": (driving(braking(17.0, 9.2, 4.5, 8.0), 0.4 * smoothstep(TEN_HZ, 9.2, 1.5)),
                                    {"PULLS": driving(102.0 + 8.0 * TEN_HZ, cuts_in(7.5, 2.5))})}
    events = scanned(tmp_path, scenes, stage1).set_index("context_name")
    nudged = events.loc["brakes_and_nudges"]
    assert (nudged["required_decel_mps2"], nudged["severity_mps2"]) == pytest.approx((3.1, 4.5), abs=0.05)
    assert nudged["own_move_m"] > 1.0 and nudged["lateral_change_mps2"] == 0.0
    ego, _ = build_ego_table(load_poses(tmp_path, SegmentRef("training", "brakes_and_nudges")), stage1.motion)
    by_entry = ego[TIME].between(nudged["onset_timestamp_micros"], nudged["entry_timestamp_micros"])
    # The pose's circle is a straight line, since the ego drives straight till 9.2 s, so each frame's pull is its own.
    assert (np.abs(scan._curvature(ego)) * ego["speed_mps"] ** 2)[by_entry].max() > 0.5
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["brakes_and_nudges", "cut_in",
                                                                                 "aborted_pass"]


def test_a_lane_change_into_a_bend_the_ego_reaches_after_onset_still_ranks_on_its_braking(tmp_path, stage1):
    """The cost of ending the window before onset, in Known risks. The road bends right at a 180 m radius from 2.9 m
    before the ego moves over behind a car that holds its lane, and both brake at 2.1 m/s² for a light. The car is on
    the bend by onset, but the ego only gets there later, so nothing says it steered, and from its pose's straight line
    the car seemed to come over by 2.2 m. It ranks above the cut-in the ego eased off for."""
    light = braking(14.0, 9.6, 2.1, 5.0)
    bend_at = np.interp(9.0, TEN_HZ, light) - 2.9
    scenes = {"bend_and_light": (driving(light, LANE * smoothstep(TEN_HZ, 9.0, 3.7), -1.0 / 180.0, bend_at),
                                 {"HOLDS": driving(light + 29.0, LANE, -1.0 / 180.0, bend_at)}),
              "cut_in": eased_off_for()}
    events = scanned(tmp_path, scenes, stage1).set_index("context_name")
    assert events.loc["bend_and_light", "lateral_change_mps2"] == 0.0
    assert events.loc["bend_and_light", "own_move_m"] == pytest.approx(2.2, abs=0.05)
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["bend_and_light", "cut_in"]


def test_an_aborted_pass_that_pulls_out_slowly_still_ranks_on_its_braking(tmp_path, stage1):
    """Another known risk. The aborted pass, but at 14 m/s, out over 7 s and back over 6 s from 11 s: no frame from its
    pose to onset pulls 0.5 m/s², and from the pose's circle the car it gave up on seemed to come over by 1.4 m. It
    ranks above the cut-in the ego eased off for."""
    out_and_back = LANE * (smoothstep(TEN_HZ, 3.0, 7.0) - smoothstep(TEN_HZ, 11.0, 6.0))
    scenes = {"slow_pass": (driving(braking(14.0, 11.0, 3.0, 8.0), out_and_back),
                            {"LEAD": driving(30.0 + braking(13.0, 11.0, 3.0, 8.5), -0.2)}),
              "cut_in": eased_off_for()}
    events = scanned(tmp_path, scenes, stage1).set_index("context_name")
    assert events.loc["slow_pass", "lateral_change_mps2"] < 0.5
    assert events.loc["slow_pass", "own_move_m"] == pytest.approx(1.4, abs=0.05)
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["slow_pass", "cut_in"]


def test_a_jolt_across_the_pose_can_lift_an_egos_lane_change_above_a_cut_in(tmp_path, stage1):
    """Another known risk. On a straight road, the ego moves over behind a car that holds its lane 29 m ahead, both
    braking at 2.1 m/s² for a light: the car's own move is 0. Onset is at 8.2 s, so the pose is at 6.2 s. A jolt 2 cm
    sideways and back, 0.4 s each way from 5.45 s to 6.25 s, across the pose, bends the pose's circle toward the car,
    and the calm frames after it pull only 0.3 m/s² off it: the car seemed to come over by 1.3 m. It ranks above the
    cut-in the ego eased off for."""
    light, over = braking(14.0, 9.6, 2.1, 5.0), LANE * smoothstep(TEN_HZ, 9.0, 3.7)
    jolt = 0.02 * (smoothstep(TEN_HZ, 5.45, 0.4) - smoothstep(TEN_HZ, 5.85, 0.4))
    holds = {"HOLDS": driving(light + 29.0, LANE)}
    scenes = {"lane_change": (driving(light, over), holds), "jolted": (driving(light, over + jolt), holds),
              "cut_in": eased_off_for()}
    events = scanned(tmp_path, scenes, stage1).set_index("context_name")
    both = events.loc[["lane_change", "jolted"]]
    assert both["onset_t_s"].tolist() == pytest.approx([8.2, 8.2])
    assert both["own_move_m"].tolist() == pytest.approx([0.0, 1.34], abs=0.05)
    assert both["lateral_change_mps2"].tolist() == pytest.approx([0.0, 0.3], abs=0.05)
    assert scan._rank(events.reset_index(), CONFIG)["context_name"].tolist() == ["jolted", "cut_in", "lane_change"]


def test_a_cut_in_that_makes_the_ego_brake_hard_ranks_first(tmp_path, stage1):
    """The earlier review's four segments: a car tucking in just past the bumper as the gap opens, a slow zipper
    merge, the ego sliding in behind a faster car, and an 8 m/s car pulling in front of a 17 m/s ego, which
    brakes at 4.5 m/s². By headway, the hard braking came last. It needed 3.1 m/s² and braked 4.5."""
    t = TEN_HZ
    events = scanned(tmp_path, {
        "tuck_in": (driving(15.0 * t), {"TUCKS": driving(19.0 * t - 25.7, cuts_in(8.0, 1.5))}),
        "zipper": (driving(3.5 * t), {"ZIPS": driving(3.5 * t + 7.8, cuts_in(8.0, 4.0))}),
        "slides_in": (driving(10.0 * t, LANE * smoothstep(t, 9.0, 3.0)), {"PASSES": driving(13.0 * t - 18.0, LANE)}),
        "brakes_hard": (driving(braking(17.0, 9.2, 4.5, 8.0)), {"PULLS": driving(102.0 + 8.0 * t, cuts_in(7.5, 2.5))}),
    }, stage1)
    assert len(events) == 4
    ranked = scan._rank(events, CONFIG)["context_name"].tolist()
    assert ranked[0] == "brakes_hard" and ranked[-1] == "slides_in"
    assert events.sort_values("headway_s")["context_name"].iloc[-1] == "brakes_hard"
    hard = events.set_index("context_name").loc["brakes_hard"]
    assert (hard["required_decel_mps2"], hard["severity_mps2"]) == pytest.approx((3.1, 4.5), abs=0.05)


def test_a_cut_in_the_ego_brakes_for_before_entry_ranks_above_mild_ones(tmp_path, stage1):
    """a4's shape: a car 3 m/s slower cuts in 6 m ahead of an 18 m/s ego's bumper, which brakes at 4.5 m/s² from
    half a second into the car's lane change, as a planner does. By entry the ego is the slower, so the gap opens
    from there on: a time to collision had nothing to measure, and ranked it below mild cut-ins nobody braked
    for. Its braking is its severity now."""
    ego = braking(18.0, 8.5, 4.5, 13.0)
    centre = np.interp(9.32, TEN_HZ, ego) + 4.0 + 6.0 + 2.3  # where the car is at entry, 6 m past the bumper
    scenes = {"brakes_early": (driving(ego), {"CUTS": driving(centre + 15.0 * (TEN_HZ - 9.32), cuts_in(8.0, 2.0))})}
    for speed in (10.0, 13.0, 16.0):  # a car 1 m/s slower moves in 20 m past the bumper, and nobody brakes
        scenes[f"mild_{speed:g}"] = (driving(speed * TEN_HZ), {"CUTS": driving(
            speed * 10.0 + 26.3 + (speed - 1.0) * (TEN_HZ - 10.0), CUT_IN)})
    events = scanned(tmp_path, scenes, stage1)
    assert events["context_name"].tolist() == ["brakes_early", "mild_10", "mild_13", "mild_16"]
    early = events.set_index("context_name").loc["brakes_early"]
    assert early["closing_mps"] < 0.0 and early["severity_mps2"] == pytest.approx(4.5, abs=0.05)
    assert scan._rank(events, CONFIG)["context_name"].iloc[0] == "brakes_early"


def test_label_noise_cannot_lift_a_same_speed_zipper_merge_above_a_hard_braking_cut_in(tmp_path, stage1):
    """a5b's shape, with a fixed seed: 4 cm of noise on each box and 2 cm on each pose, so label speeds are noisy
    too. Two cars at 3.5 m/s zip together 1.5 m past the bumper, and noise alone makes that gap seem to close now
    and then. As a time to collision that was a number, and it beat a cut-in the ego braked for before entry,
    which had none. As braking needed, it's near 0."""
    rng = np.random.default_rng(0)

    def noisy(scene, sigma):
        x, y, yaw = scene
        return x + rng.normal(0.0, sigma, x.shape), y + rng.normal(0.0, sigma, y.shape), yaw

    events = scanned(tmp_path, {
        "zipper": (noisy(driving(3.5 * TEN_HZ), 0.02),
                   {"ZIPS": noisy(driving(3.5 * TEN_HZ + 7.8, cuts_in(8.0, 4.0)), 0.04)}),
        "brakes_early": (noisy(driving(braking(15.0, 9.15, 3.0, 11.0)), 0.02),
                         {"CUTS": noisy(driving(165.3 + 13.0 * (TEN_HZ - 10.0), cuts_in(8.35, 2.5)), 0.04)}),
    }, stage1)
    zipper = events.set_index("context_name").loc["zipper"]
    assert 0.0 < zipper["required_decel_mps2"] < 0.5
    assert scan._rank(events, CONFIG)["context_name"].tolist() == ["brakes_early", "zipper"]


# The end of the segment.

def queue_stop() -> tuple:
    """The ego at 8 m/s, and a car cutting in 6 m ahead of its bumper by 10 s that then slows evenly to a stop.

    The ego brakes at 4.5 m/s² from 10.3 s and stops 2 m behind it, short of where the car entered. Car centers
    sit 6.3 m ahead of the ego's rear axle at a gap of 0: 4 m to its bumper and half a car.
    """
    t = TEN_HZ
    braking = np.clip(t - 10.3, 0.0, 8.0 / 4.5)
    ego = 8.0 * np.minimum(t, 10.3) + 8.0 * braking - 2.25 * braking**2
    entered, stopped = 80.0 + 6.0 + 6.3, ego[-1] + 2.0 + 6.3
    slowing = 8.0**2 / (2.0 * (stopped - entered))
    after = np.clip(t - 10.0, 0.0, 8.0 / slowing)
    car = entered + 8.0 * np.minimum(t - 10.0, 0.0) + 8.0 * after - slowing / 2.0 * after**2
    return driving(ego), {"QUEUES": driving(car, -LANE * (1.0 - smoothstep(t, 8.68, 2.0)))}


def test_a_cut_in_that_makes_the_ego_stop_is_found_with_the_path_end_allowance(tmp_path, stage1):
    ref = write_scene(tmp_path, "queue", *queue_stop())
    [event] = scan_segment(tmp_path, ref, stage1, CONFIG)[0].itertuples()
    assert event.past_path_end and event.own_move_m > 1.0 and event.required_decel_mps2 > 0.0
    assert scan_segment(tmp_path, ref, stage1, replace(CONFIG, path_end_allowance_m=0.0))[0].empty


def test_a_cut_in_entering_further_past_the_path_end_than_the_allowance_is_not_found(tmp_path, stage1):
    """In the segment's last half second, 12 m ahead of the bumper of an ego still at 10 m/s, 13 m past its end."""
    late = {"LATE": driving(10.0 * TEN_HZ + 18.3, -LANE * (1.0 - smoothstep(TEN_HZ, 18.0, 2.0)))}
    ref = write_scene(tmp_path, "late", driving(10.0 * TEN_HZ), late)
    assert scan_segment(tmp_path, ref, stage1, CONFIG)[0].empty
    assert len(scan_segment(tmp_path, ref, stage1, replace(CONFIG, path_end_allowance_m=14.0))[0]) == 1


# Traffic FRONT_LEFT shows whole, at speed.

def left_sample(ahead=12.0, left=2 * LANE, ego_speed=25.0, box_type="vehicle", heading=0.0) -> tuple:
    """A tracks row for one car this far ahead of the rear axle and left of it, and the ego's speed then."""
    tracks = pd.DataFrame({TIME: [T0], "type": [box_type], "center_x_m": [ahead], "center_y_m": [left],
                           "center_z_m": [0.8], "size_x_m": [VEHICLE[0]], "size_y_m": [VEHICLE[1]],
                           "size_z_m": [VEHICLE[2]], "heading_rad": [heading]})
    return tracks, pd.DataFrame({TIME: [T0], "speed_mps": [ego_speed]})


def left_count(config=CONFIG, margin_px=2.0, **kwargs) -> int:
    return scan._left_at_speed(*left_sample(**kwargs), FRONT_LEFT_CAM, margin_px, config)


def in_front_left(ahead: float, left: float, heading=0.0) -> tuple[float, float, float, float]:
    """Where FRONT_LEFT shows that car's box, unclamped: (u_min, u_max, v_min, v_max)."""
    corners = box_corners([ahead, left, 0.8], VEHICLE, heading)
    return box_rectangle(transform_points(FRONT_LEFT_CAM.camera_from_vehicle, corners), FRONT_LEFT_CAM)


def test_a_car_two_lanes_left_that_front_left_shows_whole_counts():
    u_min, u_max, v_min, v_max = in_front_left(12.0, 2 * LANE)
    assert 2.0 <= u_min and u_max <= 1918.0 and 2.0 <= v_min and v_max <= 1278.0
    assert left_count(ahead=12.0, left=2 * LANE) == 1


def test_a_car_in_the_next_lane_cut_off_at_the_bottom_of_the_image_does_not_count():
    """Check 3 can't measure a box the border cuts off. On a FRONT_LEFT turned 45 degrees, a car in the next lane
    is too close to fit: this one runs off the bottom, and off the right too."""
    assert in_front_left(8.0, LANE)[3] > FRONT_LEFT_CAM.height
    assert left_count(ahead=8.0, left=LANE) == 0


@pytest.mark.parametrize("ahead, heading, counted", [
    (8.0, 0.0, 1), (7.0, 0.0, 0),
    (7.0, -0.3, 1),  # turned toward the ego's lane, it fits a little nearer
    (7.4, 0.0, 1), (7.4, 0.3, 0),  # turned away, it needs to be further ahead
])
def test_a_car_two_lanes_left_counts_only_until_the_bottom_cuts_it_off(ahead, heading, counted):
    """Each corner goes through the calibration where the box's heading puts it."""
    u_min, u_max, _, v_max = in_front_left(ahead, 2 * LANE, heading)
    assert 2.0 <= u_min and u_max <= 1918.0 and (v_max > 1278.0) == (not counted)  # only the bottom decides
    assert left_count(ahead=ahead, heading=heading) == counted


def test_a_box_must_sit_margin_px_inside_the_image():
    """A box counts only if check 3 would measure it: margin_px or more inside the image on every side."""
    u_min, u_max, v_min, v_max = in_front_left(8.0, 2 * LANE)
    room = min(u_min, 1920.0 - u_max, v_min, 1280.0 - v_max)
    assert (left_count(ahead=8.0, margin_px=room), left_count(ahead=8.0, margin_px=room + 0.01)) == (1, 0)


@pytest.mark.parametrize("config, ego_speed, counted", [
    (CONFIG, 20.0, 1),
    (CONFIG, 19.99, 0),
    (replace(CONFIG, fast_ego_speed_mps=19.99), 19.99, 1),
])
def test_only_boxes_taken_while_the_ego_is_fast_count(config, ego_speed, counted):
    assert left_count(config, ego_speed=ego_speed) == counted


def test_only_vehicles_count_toward_left_at_speed():
    assert left_count(box_type="pedestrian") == 0


def test_without_a_calibration_nothing_counts():
    assert scan._left_at_speed(*left_sample(), None, 2.0, CONFIG) == 0


def highway(root, name: str, cars: dict, cameras=("FRONT", "FRONT_LEFT")) -> SegmentRef:
    """3 s at 25 m/s, each car holding station (ahead, left) of the rear axle, and calibration for these cameras."""
    t = np.arange(30) / 10.0
    ref = write_scene(root, name, driving(25.0 * t), {car: driving(25.0 * t + ahead, left)
                                                      for car, (ahead, left) in cars.items()})
    if cameras:
        write_calibration(root, ref, cameras)
    return ref


@pytest.mark.parametrize("cameras, counted, calibrated", [
    (("FRONT", "FRONT_LEFT"), 30, True),  # a sample a frame
    (("FRONT",), 0, False),  # a calibration with no FRONT_LEFT row
    ((), 0, False),
])
def test_scan_segment_counts_through_the_segments_own_front_left(tmp_path, stage1, cameras, counted, calibrated):
    ref = highway(tmp_path, "far_lane", {"FAR": (12.0, 2 * LANE)}, cameras=cameras)
    summary = scan_segment(tmp_path, ref, stage1, CONFIG)[1]
    assert (summary["left_at_speed"], summary["front_left_calibrated"]) == (counted, calibrated)


def test_scan_segment_takes_the_margin_from_stage1s_config(tmp_path, stage1):
    """The car 8 m ahead sits 51 px above FRONT_LEFT's bottom edge."""
    ref = highway(tmp_path, "far_lane", {"FAR": (8.0, 2 * LANE)})
    wide = replace(stage1, checks=replace(stage1.checks, border_margin_px=60.0))
    assert [scan_segment(tmp_path, ref, config, CONFIG)[1]["left_at_speed"] for config in (stage1, wide)] == [30, 0]


def test_only_the_frames_where_the_ego_is_fast_count_not_the_whole_segment(tmp_path, stage1):
    """The ego speeds up from 15 m/s at 2.4 m/s², passing 20 m/s just after 2.08 s, with a car two lanes left holding
    12 m ahead of it. Its boxes from 2.1 s on count, 19 of the 40."""
    t = np.arange(40) / 10.0
    ego = 15.0 * t + 1.2 * t * t
    ref = write_scene(tmp_path, "speeds_up", driving(ego), {"FAR": driving(ego + 12.0, 2 * LANE)})
    write_calibration(tmp_path, ref)
    assert scan_segment(tmp_path, ref, stage1, CONFIG)[1]["left_at_speed"] == 19


def test_fast_left_lists_the_three_busiest_segments_that_have_any():
    segments = pd.DataFrame({"split": "training", "context_name": list("abcdef"), "left_at_speed": [0, 5, 9, 9, 3, 1]})
    assert scan._fast_left(segments)["context_name"].tolist() == ["c", "d", "b"]  # names break the tie
    assert scan._fast_left(segments.assign(left_at_speed=[0, 0, 2, 0, 0, 0]))["context_name"].tolist() == ["c"]


@pytest.fixture(scope="module")
def fast_left_run(tmp_path_factory) -> SimpleNamespace:
    """Four highway segments at 25 m/s, through the scan job's command. busy has two cars two lanes left, light one
    and a car in the next lane, right_only a car on the right. uncalibrated has a car two lanes left for 8 s and a
    cut-in, but no camera_calibration."""
    base = tmp_path_factory.mktemp("fast_left")
    root = base / "root"
    highway(root, PRIVATE["busy"], {"FAR": (12.0, 2 * LANE), "FARTHER": (10.0, 2 * LANE)})
    highway(root, PRIVATE["light"], {"FAR": (12.0, 2 * LANE), "NEXT": (8.0, LANE)})
    highway(root, PRIVATE["right_only"], {"RIGHT": (8.0, -LANE)})
    t = np.arange(80) / 10.0
    write_scene(root, PRIVATE["uncalibrated"], driving(25.0 * t), {
        "FAR": driving(25.0 * t + 12.0, 2 * LANE),
        "CUTS": driving(118.3 + 23.0 * (t - 4.0), -LANE * (1.0 - smoothstep(t, 2.5, 2.0)))})
    outputs, home = base / "outputs", base / "home"
    result = scan_cli(*CONFIG_ARGS, env=wod_env(root, outputs, home))
    return SimpleNamespace(result=result, home=home, report_text=(outputs / "reports" / "scan.json").read_text(),
                           events=pd.read_parquet(outputs / "scan" / "events.parquet"),
                           segments=pd.read_parquet(outputs / "scan" / "segments.parquet"))


def test_fast_left_txt_lists_segments_with_cars_front_left_shows_whole_at_speed(fast_left_run):
    run = fast_left_run
    assert run.result.returncode == 0 and run.result.stderr == ""
    assert run.result.stdout.splitlines()[0] == ("segments: 4 scanned, 0 failed; events: 1 in 1 segments; top: 1, "
                                                 "steering test off: 0; fast left: 2")
    listing = run.home / "selected" / "fast_left.txt"
    assert listing.read_text() == f"{PRIVATE['busy']}\n{PRIVATE['light']}\n"
    assert stat.S_IMODE(listing.stat().st_mode) == 0o600
    counts = run.segments.set_index("context_name")["left_at_speed"].to_dict()
    assert counts == {PRIVATE["busy"]: 60, PRIVATE["light"]: 30, PRIVATE["right_only"]: 0, PRIVATE["uncalibrated"]: 0}
    printed = run.report_text + run.result.stdout + run.result.stderr
    for name in ("busy", "light", "right_only", "uncalibrated"):
        assert PRIVATE[name] not in printed and context_label(PRIVATE[name]) not in printed


def test_a_segment_without_calibration_keeps_its_cut_in_and_is_counted(fast_left_run):
    run = fast_left_run
    assert run.events["context_name"].tolist() == [PRIVATE["uncalibrated"]]
    calibrated = run.segments.set_index("context_name")["front_left_calibrated"].to_dict()
    assert calibrated == {PRIVATE["busy"]: True, PRIVATE["light"]: True, PRIVATE["right_only"]: True,
                          PRIVATE["uncalibrated"]: False}
    segments = json.loads(run.report_text)["sections"]["segments"]
    assert (segments["with_left_at_speed"], segments["without_calibration"]) == (2, 1)


# Stage 1's tables against the scan's.

@pytest.fixture(scope="module")
def private_run(tmp_path_factory) -> SimpleNamespace:
    """The synthetic scene under a private name, through Stage 1 with three cameras, as the review job runs it."""
    base = tmp_path_factory.mktemp("private")
    root = base / "root"
    ref = write_synthetic_segment(root, context_name=PRIVATE["near"])
    write_stats(root, ref)
    config = load_config(CONFIG_PATH, root=root, outputs=base / "outputs",
                         cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
    directory, _ = process_segment(config, ref)
    segment = load_segment(directory)
    return SimpleNamespace(base=base, root=root, ref=ref, config=config, segment=segment,
                           events=find_cut_ins(segment.tracks, segment.ego, CONFIG))


def test_stage1_tables_and_the_scans_own_give_the_same_events(private_run):
    """Stage 1's tracks.parquet has real synced and projected boxes, and the scan's none. Neither moves an event."""
    events, _ = scan_segment(private_run.root, private_run.ref, private_run.config, CONFIG)
    assert len(private_run.events) == 1
    assert private_run.segment.tracks["synced_center_x_m"].notna().any()
    pd.testing.assert_frame_equal(private_run.events, events.drop(columns=["split", "context_name"]))


# The scan CLI, in a real process.

@pytest.fixture(scope="module")
def scan_root(synthetic_root, tmp_path_factory) -> Path:
    """Four segments made from the synthetic one's labels, under private names, with stats.

    near has its cut-in 4 m closer, so it ranks first. far is as written. quiet has no cut-in at
    all, in the other split. broken's lidar_box isn't Parquet.
    """
    source, ref = synthetic_root
    root = tmp_path_factory.mktemp("scan_root")
    boxes = pq.read_table(component_path(source, ref, LIDAR_BOX)).to_pandas()
    cut_in = boxes[OBJECT] == OBJECT_IDS["cut_in"]
    closer = boxes.assign(**{col(LIDAR_BOX, "box.center.x"): boxes[col(LIDAR_BOX, "box.center.x")] - 4.0 * cut_in})
    segments = {
        "near": ("training", closer, ("Night", "location_sf", "rain")),
        "far": ("training", boxes, CONDITIONS),
        "quiet": ("validation", boxes[~cut_in], ("Dawn/Dusk", "location_sf", "fog")),
        "broken": ("training", boxes, CONDITIONS),
    }
    for name, (split, labels, conditions) in segments.items():
        private = SegmentRef(split, PRIVATE[name])
        for tag in (VEHICLE_POSE, LIDAR_BOX):
            component_path(root, private, tag).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(component_path(source, ref, VEHICLE_POSE), component_path(root, private, VEHICLE_POSE))
        pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), component_path(root, private, LIDAR_BOX))
        write_stats(root, private, conditions)
    component_path(root, SegmentRef("training", PRIVATE["broken"]), LIDAR_BOX).write_bytes(b"not a parquet file")
    return root


@pytest.fixture(scope="module")
def cli_run(scan_root, tmp_path_factory) -> SimpleNamespace:
    """The scan job's command, with two workers, over outputs that hold an older run's report and review page.

    Beside that page sits a file of Andrew's own, as his Stage 1 contact sheets do.
    """
    base = tmp_path_factory.mktemp("scan_run")
    outputs, home = base / "outputs", base / "home"
    for old in (outputs / "reports" / "scan.json", home / "review" / "cutins" / "cutin_1.html",
                home / "review" / "contact_sheet.html"):
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("from an older run")
    result = scan_cli(*CONFIG_ARGS, "--workers", "2", "--top", "10", env=wod_env(scan_root, outputs, home))
    report_path = outputs / "reports" / "scan.json"
    return SimpleNamespace(result=result, outputs=outputs, home=home, report_path=report_path,
                           report_text=report_path.read_text(), report=json.loads(report_path.read_text()),
                           events=pd.read_parquet(outputs / "scan" / "events.parquet"),
                           segments=pd.read_parquet(outputs / "scan" / "segments.parquet"))


def test_the_scan_prints_only_counts_and_the_report_path(cli_run):
    assert cli_run.result.returncode == 0 and cli_run.result.stderr == ""
    assert cli_run.result.stdout.splitlines() == ["segments: 3 scanned, 1 failed; events: 2 in 2 segments; top: 2, "
                                                  "steering test off: 0; fast left: 0",
                                                  f"report: {cli_run.report_path}"]


def test_a_failing_segment_is_counted_and_the_run_goes_on(cli_run):
    """pyarrow's message names the file, and so the context: only the type is kept. None of these segments has a
    camera_calibration, and each is counted."""
    assert cli_run.report["sections"]["segments"] == {"scanned": 3, "failed": 1, "failed_by_type": {"arrow_invalid": 1},
                                                      "with_left_at_speed": 0, "without_calibration": 3}
    assert cli_run.report["failures"] == []
    assert PRIVATE["broken"] not in set(cli_run.segments["context_name"])


def test_segments_rank_by_their_most_severe_event(cli_run):
    """Neither gap closes and both egos brake alike, so headway decides: near's cut-in is 4 m closer."""
    listing = cli_run.home / "selected" / "cutin_top.txt"
    assert listing.read_text() == f"{PRIVATE['near']}\n{PRIVATE['far']}\n"
    assert stat.S_IMODE(listing.stat().st_mode) == 0o600
    best = cli_run.events.groupby("context_name")["headway_s"].min()
    assert best[PRIVATE["near"]] < best[PRIVATE["far"]]
    assert (cli_run.home / "selected" / "fast_left.txt").read_text() == ""  # no ego here reaches 20 m/s


def test_events_and_segments_tables_hold_the_spec_columns(cli_run):
    events, segments = cli_run.events, cli_run.segments
    assert list(events.columns) == EVENT_COLUMNS and len(events) == 2
    assert set(events["laser_object_id"]) == {OBJECT_IDS["cut_in"]}
    assert events["onset_timestamp_micros"].dtype == np.int64 and events["entry_timestamp_micros"].dtype == np.int64
    assert list(segments.columns) == SEGMENT_COLUMNS
    rows = segments.set_index("context_name")
    assert rows.loc[PRIVATE["quiet"], ["split", "events"]].tolist() == ["validation", 0]
    assert np.isnan(rows.loc[PRIVATE["quiet"], "best_headway_s"])
    assert (rows["n_frames"] == 199).all() and rows.loc[PRIVATE["near"], "weather"] == "rain"


def test_scan_json_holds_the_spec_aggregates(cli_run):
    sections = cli_run.report["sections"]
    # Both cut-ins moved themselves, and each entered where the ego later drove. Two events are fewer than a
    # spread needs, so each shows its count alone.
    assert sections["events"] == {"events": 2, "segments_with_events": 2, "car_moved": 2, "ego_steered": 0,
                                  "past_path_end": 0,
                                  **dict.fromkeys(["headway_s", "required_decel_mps2", "ego_min_a_long_mps2",
                                                   "severity_mps2", "crossing_s", "lateral_change_mps2"],
                                                  {"count": 2})}
    assert sections["conditions"] == {"time_of_day": {"night": 1, "day": 1, "dawn_dusk": 1},
                                      "weather": {"rain": 1, "sunny": 1, "other": 1}}
    assert sections["top"] == {"segments": 2, "steering_test_off": 0, "daytime": 1}
    assert sections["config"] == asdict(CONFIG)


@pytest.fixture(scope="module")
def queue_runs(tmp_path_factory) -> list[tuple[subprocess.CompletedProcess, dict, pd.DataFrame]]:
    """Three queue stops through the scan job's command, the third missing a second of frames between its pose and
    the window's end, so its speed and yaw rate are unknown there. Then again with min_own_move_m above their own
    moves, and with max_lateral_change_mps2 below 0, so that even an ego driving straight has steered.

    Each run, its scan.json's events section and its events.
    """
    base = tmp_path_factory.mktemp("queue")
    for name, lost in (("near", None), ("far", None), ("broken", (7.55, 8.55))):
        write_scene(base / "root", PRIVATE[name], *queue_stop(), lost=lost)
    tomls = [SCAN_CONFIG_PATH]
    for key, value in (("min_own_move_m", 9.0), ("max_lateral_change_mps2", -1.0)):
        tomls.append(base / f"{key}.toml")
        tomls[-1].write_text(re.sub(rf"^{key} = \S+", f"{key} = {value}", SCAN_CONFIG_PATH.read_text(), flags=re.M))
    runs = []
    for run, scan_config in enumerate(tomls):
        result = scan_cli("--config", str(CONFIG_PATH), "--scan-config", str(scan_config),
                          env=wod_env(base / "root", base / f"outputs_{run}", base / f"home_{run}"))
        report = json.loads((base / f"outputs_{run}" / "reports" / "scan.json").read_text())
        runs.append((result, report["sections"]["events"], pd.read_parquet(base / f"outputs_{run}" / "scan" /
                                                                         "events.parquet")))
    return runs


def test_scan_json_counts_the_entries_past_the_path_end_and_the_cars_that_moved(queue_runs):
    """Each queue stop's entry needs the path-end allowance, and its car moved itself. Where the frames are missing,
    the sideways pull is unknown, so nothing says the ego kept to its circle: steered, though its own move is known."""
    (result, events, table), (_, strict, _), (_, steered, steered_table) = queue_runs
    assert result.returncode == 0 and result.stderr == ""
    assert events == {"events": 3, "segments_with_events": 3, "car_moved": 2, "ego_steered": 1, "past_path_end": 3,
                      **dict.fromkeys(["headway_s", "required_decel_mps2", "ego_min_a_long_mps2", "severity_mps2",
                                       "crossing_s"], {"count": 3}), "lateral_change_mps2": {"count": 2}}
    assert (table["own_move_m"] > 1.0).all() and table["lateral_change_mps2"].isna().sum() == 1
    # The line comes from the config. Under it, an own move is known and small: ranked last, but not steered.
    assert (strict["events"], strict["car_moved"], strict["ego_steered"]) == (3, 0, 1)
    assert (steered["events"], steered["car_moved"], steered["ego_steered"]) == (3, 0, 3)  # every pull past the line
    assert steered["lateral_change_mps2"] == {"count": 2}  # the spread still counts the steered events
    pd.testing.assert_frame_equal(steered_table, table)  # the limit only weighs the own move: it never changes it


@pytest.fixture(scope="module")
def severity_run(tmp_path_factory) -> SimpleNamespace:
    """Ten segments through the scan job's command, enough for a spread to show its numbers. In each, an 8 m/s car
    pulls in front of a 17 m/s ego, which brakes harder each time, from 3.3 m/s² to 6.0. Half have a calibration, and
    of the rest, three have one that can't be read."""
    base = tmp_path_factory.mktemp("severity")
    names = [f"{'9' * 19}{i}_100_000_120_000" for i in range(10)]
    for i, name in enumerate(names):
        ref = write_scene(base / "root", name, driving(braking(17.0, 9.2, 3.3 + 0.3 * i, 8.0)),
                          {"PULLS": driving(102.0 + 8.0 * TEN_HZ, cuts_in(7.5, 2.5))})
        if i % 2:
            write_calibration(base / "root", ref)
        elif i < 6:
            write_unreadable_calibration(base / "root", ref, ("empty", "cut_short", "no_width")[i // 2])
    result = scan_cli(*CONFIG_ARGS, env=wod_env(base / "root", base / "outputs", base / "home"))
    report_text = (base / "outputs" / "reports" / "scan.json").read_text()
    return SimpleNamespace(result=result, names=names, report_text=report_text,
                           sections=json.loads(report_text)["sections"])


def test_scan_json_spreads_the_severity_and_counts_segments_without_calibration(severity_run):
    """Each needed about 3.1 m/s² as the ego started braking, and braked harder than that. A calibration that can't be
    read costs only left_at_speed, as a missing one does: every segment keeps its cut-in."""
    sections = severity_run.sections
    assert severity_run.result.returncode == 0 and severity_run.result.stderr == ""
    assert (sections["segments"]["scanned"], sections["segments"]["failed"]) == (10, 0)
    assert sections["segments"]["without_calibration"] == 5
    needed, severity = sections["events"]["required_decel_mps2"], sections["events"]["severity_mps2"]
    assert needed["count"] == severity["count"] == 10
    assert (needed["min"], needed["max"]) == pytest.approx((3.1, 3.1), abs=0.1)
    assert (severity["min"], severity["median"], severity["max"]) == pytest.approx((3.3, 4.65, 6.0), abs=0.05)
    # Every ego drives straight, so the low end of this spread is where wander and yaw noise would show on real poses.
    assert sections["events"]["lateral_change_mps2"] == {"count": 10, **dict.fromkeys(scan._QUANTILES, 0.0)}
    printed = severity_run.report_text + severity_run.result.stdout + severity_run.result.stderr
    assert not any(name in printed or context_label(name) in printed for name in severity_run.names)


def test_scan_json_passes_report_add_and_names_nothing(cli_run):
    report = Report("check", cli_run.report["preprocessing_version"])
    for name, section in cli_run.report["sections"].items():
        report.add(name, **section)  # raises on any name, string, or number the boundary refuses
    values = leaves(cli_run.report["sections"])
    assert {value for value in values if isinstance(value, str)} <= VOCABULARY
    assert max(abs(value) for value in values if isinstance(value, (int, float))) < 1e12
    printed = cli_run.report_text + cli_run.result.stdout + cli_run.result.stderr
    for name in PRIVATE.values():
        assert name not in printed and context_label(name) not in printed  # a list of labels is a context list
    assert not any(object_id in printed for object_id in OBJECT_IDS.values())


def test_scan_json_holds_no_events_own_values(cli_run):
    """Two events are fewer than a spread needs, so every number but the config's is a count. An event's own
    value would be a float, as JSON writes one even when it's whole."""
    sections = {name: section for name, section in cli_run.report["sections"].items() if name != "config"}
    assert all(type(value) is int for value in leaves(sections))


def test_a_spread_shows_its_numbers_only_from_ten_events():
    """Each of its five numbers can be one event's own value: from five events, they'd be all of them."""
    assert scan._spread(pd.Series([5.0, 3.0, 1.0, 2.0, 4.0, np.nan])) == {"count": 5}
    assert scan._spread(pd.Series(np.arange(9.0))) == {"count": 9}
    assert scan._spread(pd.Series([*np.arange(10.0)[::-1], np.nan])) == {
        "count": 10, "min": 0.0, "p25": 2.25, "median": 4.5, "p75": 6.75, "max": 9.0}


def test_a_new_scan_clears_the_cut_in_pages_and_nothing_else_in_review(cli_run):
    """Pages go by rank, and the ranks just changed. Andrew's own files in review stay."""
    assert not (cli_run.home / "review" / "cutins").exists()
    assert (cli_run.home / "review" / "contact_sheet.html").read_text() == "from an older run"


def test_what_the_scan_writes_is_private(cli_run):
    """env.sh sets umask 077, but the run shouldn't lean on it: the tables and the lists name contexts."""
    scanned_dir = cli_run.outputs / "scan"
    paths = [scanned_dir, *scanned_dir.iterdir(), cli_run.report_path, *(cli_run.home / "selected").rglob("*")]
    assert len(paths) == 6 and not [path for path in paths if stat.S_IMODE(path.stat().st_mode) & 0o077]


# The scan CLI, in this process.

@pytest.mark.parametrize("args, counts, listed", [
    (["--limit", "2"], "segments: 2 scanned, 0 failed; events: 2 in 2 segments; top: 2, steering test off: 0; "
                       "fast left: 0",
     ["near", "far"]),  # by name
    (["--splits", "validation"], "segments: 1 scanned, 0 failed; events: 0 in 0 segments; top: 0, steering test off: "
                                 "0; fast left: 0", []),
    (["--top", "1"], "segments: 3 scanned, 1 failed; events: 2 in 2 segments; top: 1, steering test off: 0; "
                     "fast left: 0", ["near"]),
])
def test_flags_choose_what_is_scanned_and_listed(scan_root, tmp_path, monkeypatch, capfd, args, counts, listed):
    for name, value in wod_env(scan_root, tmp_path / "outputs", tmp_path / "home").items():
        monkeypatch.setenv(name, value)
    assert scan.main([*CONFIG_ARGS, *args]) == 0
    assert capfd.readouterr().out.splitlines()[0] == counts
    listing = tmp_path / "home" / "selected" / "cutin_top.txt"
    assert listing.read_text().splitlines() == [PRIVATE[name] for name in listed]


def scan_of(tmp_path, monkeypatch, segments: dict[str, list[dict]], *args) -> SimpleNamespace:
    """main, in this process, over segments that hold these events: each lane_change's own event with these columns.

    _scan_all is replaced, so no segment is read: what's tested is what the scan makes of its events. Returns
    cutin_top.txt's names and scan.json's sections.
    """
    base = find_cut_ins(*lane_change(), CONFIG).assign(split="training")
    results = [(pd.concat([base.assign(context_name=name, **row) for row in rows], ignore_index=True),
                {**dict.fromkeys(scan._SEGMENT_COLUMNS, 0), "split": "training", "context_name": name,
                 "events": len(rows), "front_left_calibrated": True, "time_of_day": "Day", "weather": "sunny",
                 "location": "location_sf"}) for name, rows in segments.items()]
    monkeypatch.setattr(scan, "_scan_all", lambda *_: results)
    for name, value in wod_env(tmp_path / "root", tmp_path / "outputs", tmp_path / "home").items():
        monkeypatch.setenv(name, value)
    assert scan.main([*CONFIG_ARGS, *args]) == 0
    report = json.loads((tmp_path / "outputs" / "reports" / "scan.json").read_text())
    return SimpleNamespace(listed=(tmp_path / "home" / "selected" / "cutin_top.txt").read_text().splitlines(),
                           sections=report["sections"])


def test_the_list_is_the_ranked_top_then_the_top_as_if_the_ego_never_steered(tmp_path, monkeypatch, capfd):
    """--top 2. Only kept_a and kept_b keep to their circles, so they are the first part. With every pull read as 0,
    unknown_pull (5.0) and pulled_hard (4.0) rank first, so they are the second part, in that order: not by name,
    and not as the first ranking has them, where pulled_hard's 9.5 event leads the ones ranked last. ego_moved's
    small own move ranks it last either way, and pulled_mildly (1.0) ranks fifth with the pulls read as 0, so
    neither is listed."""
    run = scan_of(tmp_path, monkeypatch, {
        "kept_b": [dict(severity_mps2=2.0)],
        "pulled_mildly": [dict(severity_mps2=1.0, lateral_change_mps2=0.7)],
        "unknown_pull": [dict(severity_mps2=5.0, lateral_change_mps2=np.nan)],
        "ego_moved": [dict(severity_mps2=9.0, own_move_m=0.5, lateral_change_mps2=0.9)],
        "pulled_hard": [dict(severity_mps2=4.0, lateral_change_mps2=0.79), dict(severity_mps2=9.5, own_move_m=0.5)],
        "kept_a": [dict(severity_mps2=3.0)],
    }, "--top", "2")
    assert run.listed == ["kept_a", "kept_b", "unknown_pull", "pulled_hard"]
    assert run.sections["top"] == {"segments": 2, "steering_test_off": 2, "daytime": 4}
    assert capfd.readouterr().out.splitlines()[0] == ("segments: 6 scanned, 0 failed; events: 7 in 6 segments; top: 2, "
                                                      "steering test off: 2; fast left: 0")


@pytest.mark.parametrize("top, listed, counts", [
    ("2", ["kept_a", "kept_b", "pulled"], (2, 1)),  # kept_a is in both tops, and listed once: kept_c is in neither
    ("1", ["kept_a", "pulled"], (1, 1)),
    ("10", ["kept_a", "kept_b", "kept_c", "pulled"], (4, 0)),  # all four rank in the first part, pulled on its mild
])
def test_the_list_names_each_segment_once_and_may_be_shorter_than_twice_top(tmp_path, monkeypatch, top, listed,
                                                                            counts):
    """A segment ranks by its most severe event in each part: pulled by its mild one first, and its steered one with
    the steering test off."""
    run = scan_of(tmp_path, monkeypatch, {"kept_b": [dict(severity_mps2=2.0)], "kept_a": [dict(severity_mps2=3.0)],
                                          "kept_c": [dict(severity_mps2=0.5)],
                                          "pulled": [dict(severity_mps2=5.0, lateral_change_mps2=0.79),
                                                     dict(severity_mps2=0.1)]}, "--top", top)
    assert run.listed == listed
    assert (run.sections["top"]["segments"], run.sections["top"]["steering_test_off"]) == counts


def test_scan_json_counts_the_cars_that_moved_and_the_egos_that_steered(tmp_path, monkeypatch):
    """car_moved: an own move of min_own_move_m or more, with the pull within the line. ego_steered: a pull past it
    or unknown, whatever the own move, as when an unknown circle leaves both unknown."""
    rows = [dict(own_move_m=1.6, lateral_change_mps2=0.5), dict(own_move_m=1.0, lateral_change_mps2=0.0),
            dict(own_move_m=0.99, lateral_change_mps2=0.0), dict(own_move_m=np.nan, lateral_change_mps2=0.0),
            dict(own_move_m=1.6, lateral_change_mps2=0.501), dict(own_move_m=0.5, lateral_change_mps2=0.9),
            dict(own_move_m=1.6, lateral_change_mps2=np.nan), dict(own_move_m=np.nan, lateral_change_mps2=np.nan)]
    events = scan_of(tmp_path / "default", monkeypatch, {"a": rows}).sections["events"]
    assert (events["events"], events["car_moved"], events["ego_steered"]) == (8, 2, 4)
    toml = tmp_path / "scan.toml"
    toml.write_text(SCAN_CONFIG_PATH.read_text().replace("max_lateral_change_mps2 = 0.5",
                                                         "max_lateral_change_mps2 = 0.501"))
    events = scan_of(tmp_path / "wider", monkeypatch, {"a": rows}, "--scan-config", str(toml)).sections["events"]
    assert (events["car_moved"], events["ego_steered"]) == (3, 3)


def test_the_scan_lists_a_steered_segment_after_the_ranked_top(tmp_path, monkeypatch, capfd):
    """Through the scan, at --top 1: the cut-in the ego eased off for is the ranked top. The aborted pass, whose
    pull-out ranks it last, comes second: with the steering test off, its braking would put it first."""
    for name, scene in {"aborted_pass": aborted_pass(), "cut_in": eased_off_for()}.items():
        write_scene(tmp_path / "root", name, *scene)
    for name, value in wod_env(tmp_path / "root", tmp_path / "outputs", tmp_path / "home").items():
        monkeypatch.setenv(name, value)
    assert scan.main([*CONFIG_ARGS, "--top", "1"]) == 0
    assert capfd.readouterr().out.splitlines()[0].endswith("top: 1, steering test off: 1; fast left: 0")
    assert (tmp_path / "home" / "selected" / "cutin_top.txt").read_text() == "cut_in\naborted_pass\n"


def test_the_help_says_the_list_holds_up_to_twice_top_and_review_pages_each(capsys):
    with pytest.raises(SystemExit):
        scan.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--top K list the ranked top K, then up to K more that rank there with the steering test off" in help_text
    assert "review: a page per listed segment, once agatha-stage1 has run on it" in help_text


def test_a_run_where_every_segment_fails_still_reports(scan_root, tmp_path, monkeypatch, capfd):
    """far's lidar_box isn't Parquet either, so two segments fail alike: the printed line and scan.json count both,
    not their one type."""
    root = tmp_path / "root"
    shutil.copytree(scan_root / "training", root / "training", ignore=shutil.ignore_patterns(f"{PRIVATE['near']}*"))
    component_path(root, SegmentRef("training", PRIVATE["far"]), LIDAR_BOX).write_bytes(b"not a parquet file")
    for name, value in wod_env(root, tmp_path / "outputs", tmp_path / "home").items():
        monkeypatch.setenv(name, value)
    assert scan.main(CONFIG_ARGS) == 0
    assert capfd.readouterr().out.splitlines()[0] == ("segments: 0 scanned, 2 failed; events: 0 in 0 segments; top: 0, "
                                                      "steering test off: 0; fast left: 0")
    report = json.loads((tmp_path / "outputs" / "reports" / "scan.json").read_text())
    assert report["sections"]["segments"] == {"scanned": 0, "failed": 2, "failed_by_type": {"arrow_invalid": 2},
                                              "with_left_at_speed": 0, "without_calibration": 0}
    events = pd.read_parquet(tmp_path / "outputs" / "scan" / "events.parquet")
    assert list(events.columns) == EVENT_COLUMNS and events.empty


def test_a_pool_worker_dies_on_sigterm():
    """Linux forks pool workers, so they'd inherit main's handler and catch a SIGTERM as a SystemExit.

    A pool broken by one worker's crash sends the rest SIGTERM, and Slurm's time limit sends it to all. Each
    worker here gets one mid-segment, and must die, breaking the pool, rather than carry on.
    """
    code = textwrap.dedent("""
        import multiprocessing, os, signal, sys
        from types import SimpleNamespace
        from agatha import scan
        multiprocessing.set_start_method("fork")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # as main installs it
        scan.scan_segment = lambda *_: os.kill(os.getpid(), signal.SIGTERM)
        try:
            scan._scan_all(["a", "b"], SimpleNamespace(root=None), None, workers=2)
        except BaseException as exc:
            print(type(exc).__name__)
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.stdout == "BrokenProcessPool\n"


@pytest.mark.parametrize("function, step", [("_scan_all", "scan"), ("write_atomic", "outputs")])
def test_a_failing_step_writes_the_report_with_no_message(scan_root, tmp_path, monkeypatch, capfd, function, step):
    message = f"{OBJECT_IDS['lead']} in {PRIVATE['near']}"

    def explode(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(scan, function, explode)
    for name, value in wod_env(scan_root, tmp_path / "outputs", tmp_path / "home").items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exit_info:
        scan.main(CONFIG_ARGS)
    out, err = capfd.readouterr()
    text = (tmp_path / "outputs" / "reports" / "scan.json").read_text()
    assert exit_info.value.code == 1
    assert OBJECT_IDS["lead"] not in out + err + text and PRIVATE["near"] not in out + err + text
    report = json.loads(text)
    assert [(failure["step"], failure["exception"]) for failure in report["failures"]] == [(step, "RuntimeError")]
    assert re.fullmatch(r"scan\.py:\d+", report["failures"][0]["location"])
    assert report["sections"]["config"] == asdict(CONFIG)  # so a failed run still says what it was asked
    assert ("segments" in report["sections"]) == (step == "outputs")  # what was counted before the failure stays


def test_a_mistyped_command_does_not_repeat_its_arguments():
    result = scan_cli(PRIVATE["near"], env={})
    assert result.returncode == 2 and result.stdout == ""
    assert PRIVATE["near"] not in result.stderr


def test_a_missing_home_fails_quietly(scan_root, tmp_path):
    """env.sh sets WOD_HOME. Without it, nothing says where the context list may go."""
    result = scan_cli(*CONFIG_ARGS, env={**wod_env(scan_root, tmp_path, tmp_path), "WOD_HOME": None})
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "agatha-scan: failed outside any step\n"
    assert not (tmp_path / "selected").exists()


# The review pages.

@pytest.fixture(scope="module")
def reviewed(private_run) -> SimpleNamespace:
    """agatha-scan review over three candidates, of which only the second has whole Stage 1 outputs.

    The first had no camera data, so Stage 1 never ran on it. The third's run was killed while
    writing its tables, which leaves them with no segment.json. Beside the pages sits a file of Andrew's.
    """
    home = private_run.base / "home"
    listing = home / "selected" / "cutin_top.txt"
    listing.parent.mkdir(parents=True)
    listing.write_text(f"{PRIVATE['far']}\n{private_run.ref.context_name}\n{PRIVATE['quiet']}\n")
    half_written = private_run.config.outputs / PRIVATE["quiet"]
    half_written.mkdir()
    shutil.copy(private_run.segment.directory / "frames.parquet", half_written)
    stale = home / "review" / "cutins" / "cutin_3.html"  # from an older list, where third was another segment
    stale.parent.mkdir(parents=True)
    stale.write_text("from an older list")
    (home / "review" / "contact_sheet.html").write_text("Andrew's")
    result = scan_cli("review", *CONFIG_ARGS, env=wod_env(private_run.root, private_run.config.outputs, home))
    page = (home / "review" / "cutins" / "cutin_2.html").read_text()
    captions = re.findall(r"<figcaption>(\w+) at ([\d.]+) s(, onset|, entry)?</figcaption>", page)
    images = [base64.b64decode(image) for image in re.findall(r'src="data:image/jpeg;base64,([^"]+)"', page)]
    return SimpleNamespace(result=result, home=home, page=page, captions=captions, images=images)


def test_review_prints_only_counts_and_where_the_pages_are(reviewed):
    assert reviewed.result.returncode == 0 and reviewed.result.stderr == ""
    assert reviewed.result.stdout.splitlines() == [
        "review: 1 pages, 1 events; 2 without Stage 1 outputs, 0 failed",
        f"pages: {reviewed.home / 'review' / 'cutins'}"]


def test_pages_go_by_rank_in_their_own_directory_and_an_older_lists_pages_go(reviewed):
    assert [path.name for path in (reviewed.home / "review" / "cutins").iterdir()] == ["cutin_2.html"]
    assert (reviewed.home / "review" / "contact_sheet.html").read_text() == "Andrew's"


def test_no_context_name_reaches_a_page_or_the_output(reviewed):
    printed = reviewed.page + reviewed.result.stdout
    for name in PRIVATE.values():
        assert name not in printed and context_label(name) not in printed


def test_each_event_shows_front_at_2_fps_from_onset_minus_1_s_to_entry_plus_2_s(reviewed, private_run):
    [event] = private_run.events.itertuples()
    assert {alias for alias, _, _ in reviewed.captions} == {event.alias}
    times = [float(t_s) for _, t_s, _ in reviewed.captions]
    assert times == sorted(times) and len(reviewed.images) == len(times)
    marked = {mark: float(t_s) for _, t_s, mark in reviewed.captions if mark}
    assert marked == {", onset": round(event.onset_t_s, 2), ", entry": round(event.entry_t_s, 2)}
    on_grid = [float(t_s) for _, t_s, mark in reviewed.captions if not mark]
    expected = np.arange(math.ceil((event.onset_t_s - 1.0) * 2.0) / 2.0, event.entry_t_s + 2.0 + 1e-6, 0.5)
    np.testing.assert_allclose(on_grid, expected, atol=0.01)


def test_each_event_has_a_line_with_its_numbers(reviewed, private_run):
    """The synthetic cut-in is faster than the ego, so it needed nothing, and the ego brakes gently throughout."""
    [event] = private_run.events.itertuples()
    assert (f"<h2>{event.alias}: headway {event.headway_s:.2f} s, needed 0.0 m/s² to avoid it, braked 0.4 m/s², "
            f"crossing {event.crossing_s:.2f} s, own move {event.own_move_m:.2f} m</h2>") in reviewed.page


SAID = "needed 3.1 m/s² to avoid it, braked 4.5 m/s², crossing 1.00 s, own move"
PULLED = ". The ego pulled 0.79 m/s² sideways before onset, so the own move may be wrong"


@pytest.mark.parametrize("needed, a_long, own_move_m, pull, config, says", [
    (3.1, -4.5, 1.6, 0.0, CONFIG, f"{SAID} 1.60 m"),
    (0.0, 0.3, 1.6, 0.0, CONFIG, "needed 0.0 m/s² to avoid it, braked 0.0 m/s², crossing 1.00 s, own move 1.60 m"),
    (0.0, np.nan, 1.6, 0.0, CONFIG, "needed 0.0 m/s² to avoid it, braked 0.0 m/s², crossing 1.00 s, own move 1.60 m"),
    (3.1, -4.5, 1.0, 0.0, CONFIG, f"{SAID} 1.00 m"),
    (3.1, -4.5, 0.99, 0.0, CONFIG, f"{SAID} 0.99 m. The ego probably moved, not the car"),
    (3.1, -4.5, 0.99, 0.0, replace(CONFIG, min_own_move_m=0.98), f"{SAID} 0.99 m"),
    (3.1, -4.5, np.nan, 0.0, CONFIG, f"{SAID} unknown"),  # nothing says who moved
    (3.1, -4.5, 1.64, 0.5, CONFIG, f"{SAID} 1.64 m"),
    (3.1, -4.5, 1.64, 0.79, CONFIG, f"{SAID} 1.64 m{PULLED}"),
    (3.1, -4.5, 1.64, 0.79, replace(CONFIG, max_lateral_change_mps2=0.79), f"{SAID} 1.64 m"),
    (3.1, -4.5, 1.64, np.nan, CONFIG, f"{SAID} 1.64 m, pull before onset unknown"),  # no steer it can name
    (3.1, -4.5, np.nan, np.nan, CONFIG, f"{SAID} unknown, pull before onset unknown"),  # the circle is unknown
    (3.1, -4.5, 0.4, np.nan, CONFIG, f"{SAID} 0.40 m, pull before onset unknown. The ego probably moved, not the car"),
    (3.1, -4.5, 0.4, 0.79, CONFIG, f"{SAID} 0.40 m. The ego probably moved, not the car{PULLED}"),
])
def test_an_events_line_gives_its_severity_and_says_when_the_ego_may_have_moved(needed, a_long, own_move_m, pull,
                                                                               config, says):
    """Braking counts as severity does: speeding up, or not knowing, is no braking. A small own move says the ego
    probably moved, and a pull past the limit, with its size, that the own move may be wrong. An unknown pull is said
    plainly, beside the own move, as it ranks the event last but claims no steer."""
    event = SimpleNamespace(alias="V001", headway_s=1.2, required_decel_mps2=needed, ego_min_a_long_mps2=a_long,
                            crossing_s=1.0, own_move_m=own_move_m, lateral_change_mps2=pull)
    assert scan._event_line(event, config) == f"V001: headway 1.20 s, {says}"


def test_the_page_says_when_the_ego_may_have_moved(private_run):
    events = private_run.events.assign(required_decel_mps2=3.1, ego_min_a_long_mps2=-4.5, own_move_m=0.4,
                                       lateral_change_mps2=0.79)
    page = scan._review_page(private_run.segment, events, private_run.config, CONFIG, 1)
    assert re.search(r"<h2>V\d+: headway [\d.]+ s, needed 3\.1 m/s² to avoid it, braked 4\.5 m/s², .* "
                     r"own move 0\.40 m\. The ego probably moved, not the car\. The ego pulled 0\.79 m/s² sideways "
                     r"before onset, so the own move may be wrong</h2>", page)


def test_review_frames_use_the_full_profile(reviewed):
    for image in reviewed.images:
        assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (1280, 1920, 3)


def test_the_cut_in_vehicle_is_drawn_thicker_on_the_page(reviewed, private_run):
    [event] = private_run.events.itertuples()
    timestamp = event.entry_timestamp_micros
    image = reviewed.images[[mark for _, _, mark in reviewed.captions].index(", entry")]
    segment, full = private_run.segment, private_run.config.profiles["full"]
    cam = camera_models(segment.meta)[FRONT]
    jpeg = read_jpegs(private_run.root, private_run.ref, FRONT, [timestamp])[timestamp]
    boxes = boxes_for_image(segment, timestamp, FRONT, cam)
    assert event.alias in {box.alias for box in boxes}
    assert image == render_annotated(jpeg, boxes, cam, full, thick_alias=event.alias)
    assert image != render_annotated(jpeg, boxes, cam, full)


def test_a_page_that_fails_is_counted_and_the_rest_go_on(private_run, tmp_path, monkeypatch, capfd):
    home = tmp_path / "home"
    (home / "selected").mkdir(parents=True)
    (home / "selected" / "cutin_top.txt").write_text(f"{private_run.ref.context_name}\n")

    def explode(*args):
        raise RuntimeError(private_run.ref.context_name)

    monkeypatch.setattr(scan, "read_jpegs", explode)
    for name, value in wod_env(private_run.root, private_run.config.outputs, home).items():
        monkeypatch.setenv(name, value)
    assert scan.main(["review", *CONFIG_ARGS]) == 0
    out, err = capfd.readouterr()
    assert out.splitlines()[0] == "review: 0 pages, 0 events; 0 without Stage 1 outputs, 1 failed"
    assert private_run.ref.context_name not in out + err


def test_render_draws_only_the_named_alias_thicker():
    pinhole = CameraModel((1000.0, 1000.0, 960.0, 640.0, 0, 0, 0, 0, 0), np.eye(4), 1920, 1280)
    jpeg = cv2.imencode(".jpg", np.full((1280, 1920, 3), 128, np.uint8))[1].tobytes()
    box = DrawBox("V001", "vehicle", box_corners((12.0, 0.0, 0.0), (4.0, 6.0, 4.0), 0.0), "lidar_box", False)
    profile = RenderProfile("test", 1920, 3, 0.4, 92)

    def blue_pixels(thick_alias) -> int:  # vehicles are drawn blue, and nothing else here is
        image = cv2.imdecode(np.frombuffer(render_annotated(jpeg, [box], pinhole, profile, thick_alias), np.uint8),
                             cv2.IMREAD_COLOR).astype(int)
        return int(((image[..., 0] > 200) & (image[..., 2] < 60)).sum())

    assert blue_pixels("V001") > 2 * blue_pixels(None)
    assert render_annotated(jpeg, [box], pinhole, profile, "V002") == render_annotated(jpeg, [box], pinhole, profile)


# The Slurm jobs, with uv and gcloud stubbed out.

STUB_UV = """#!/bin/bash
echo "$*" >> "$UV_LOG"
cat > /dev/null  # as ffmpeg does, inside agatha-stage1
echo "uv $*"  # names the context, as a label in a report path would
echo "uv $*" >&2
[[ "$*" != *"--split validation"* ]]  # Stage 1 fails on the validation segment
"""
STUB_GCLOUD = """#!/bin/bash
echo "$*" >> "$GCLOUD_LOG"
cat > /dev/null  # as anything reading stdin would
echo "gcloud $*"  # names the object, as its own errors do
echo "gcloud $*" >&2
if [ "$2" = rsync ]; then  # copies 4, 3, 2 or 1 of $GCLOUD_OBJECTS by component into its last argument, naming each
  case "$(basename "${@: -1}")" in lidar_box) n=4 ;; vehicle_pose) n=3 ;; stats) n=2 ;; *) n=1 ;; esac
  for name in $GCLOUD_OBJECTS; do
    [ "$n" -gt 0 ] || break
    n=$((n - 1)); touch "${@: -1}/$name.parquet"; echo "Copying $name.parquet" >&2
  done
  touch "${@: -1}/_metadata" "${@: -1}/part.parquet_.gstmp"  # the bucket's entry, and a file still being written
fi
[[ "$*" != *"$GCLOUD_FAILS"* ]] || exit 2
"""
STAGE1 = ("run --frozen agatha-stage1 --config configs/stage1.toml --split {} --context {} "
          "--cameras FRONT FRONT_LEFT FRONT_RIGHT")
REVIEW = "run --frozen agatha-scan review --config configs/stage1.toml --scan-config configs/scan.toml"
CAMERA_COMPONENTS = ("camera_image", "camera_calibration", "lidar_camera_synced_box", "projected_lidar_box")
LABEL_COMPONENTS = ("lidar_box", "vehicle_pose", "stats", "camera_calibration")


def run_job(job: str, tmp_path: Path, *args: str, listed=(), list_name="cutin_top.txt", files=(),
            fails="no such object", objects=()) -> tuple[subprocess.CompletedProcess, list[str], list[str]]:
    """A job script run from a directory whose infra/pace/env.sh points at test directories.

    `files` are (split, component, context) to make under WOD_ROOT, and gcloud fails on any object
    whose path holds `fails`. gcloud storage rsync copies `objects` by name. Returns the run and the
    arguments of each uv and gcloud call.
    """
    home = tmp_path / "home"
    (tmp_path / "infra" / "pace").mkdir(parents=True)
    (tmp_path / "infra" / "pace" / "env.sh").write_text(f'export WOD_HOME="{home}"\nexport WOD_ROOT="{home}/root"\n')
    (tmp_path / "bin").mkdir()
    for stub, text in (("uv", STUB_UV), ("gcloud", STUB_GCLOUD)):
        (tmp_path / "bin" / stub).write_text(text)
        (tmp_path / "bin" / stub).chmod(0o755)
        (tmp_path / f"{stub}.log").touch()
    (home / "selected").mkdir(parents=True)
    (home / "selected" / list_name).write_text("".join(f"{context}\n" for context in listed))
    for split, component, context in files:
        path = home / "root" / split / component / f"{context}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "UV_LOG": str(tmp_path / "uv.log"),
           "GCLOUD_LOG": str(tmp_path / "gcloud.log"), "GCLOUD_FAILS": fails, "GCLOUD_OBJECTS": " ".join(objects)}
    result = subprocess.run(["bash", str(REPO / "infra" / "pace" / job), *args], cwd=tmp_path, env=env,
                            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    return result, *((tmp_path / f"{stub}.log").read_text().splitlines() for stub in ("uv", "gcloud"))


def test_the_review_job_runs_stage1_by_rank_and_never_prints_a_name(tmp_path):
    contexts = [PRIVATE["near"], PRIVATE["far"], PRIVATE["quiet"]]
    result, calls, _ = run_job("review_cutins.sbatch", tmp_path, "cutin_top.txt", listed=contexts,
                               files=[("training", "camera_image", PRIVATE["near"]),
                                      ("validation", "camera_image", PRIVATE["quiet"])])
    assert result.returncode == 0
    assert calls == [STAGE1.format("training", PRIVATE["near"]), STAGE1.format("validation", PRIVATE["quiet"]), REVIEW]
    assert result.stdout.splitlines() == [
        "cutin_1: agatha-stage1 exit 0", "cutin_3: agatha-stage1 exit 1", f"uv {REVIEW}"]
    assert not any(name in result.stdout + result.stderr for name in contexts)


def test_the_review_job_runs_stage1_on_fast_left_segments_with_no_pages(tmp_path):
    contexts = [PRIVATE["quiet"], PRIVATE["near"]]
    result, calls, _ = run_job("review_cutins.sbatch", tmp_path, "fast_left.txt", listed=contexts,
                               list_name="fast_left.txt", files=[("training", "camera_image", PRIVATE["near"]),
                                                                 ("validation", "camera_image", PRIVATE["quiet"])])
    assert result.returncode == 0
    assert calls == [STAGE1.format("validation", PRIVATE["quiet"]), STAGE1.format("training", PRIVATE["near"])]
    assert result.stdout.splitlines() == ["fast_left_1: agatha-stage1 exit 1", "fast_left_2: agatha-stage1 exit 0"]
    assert not any(name in result.stdout + result.stderr for name in contexts)


def test_download_listed_fetches_camera_data_by_line_and_never_prints_a_name(tmp_path):
    """Each segment's split is the one holding its labels. far has none, and gcloud fails on one of broken's
    files, printing its name as gcloud does. Neither stops the run."""
    contexts = [PRIVATE["near"], PRIVATE["broken"], PRIVATE["far"], PRIVATE["quiet"]]
    labeled = [("training", PRIVATE["near"]), ("training", PRIVATE["broken"]), ("validation", PRIVATE["quiet"])]
    result, _, calls = run_job("download_listed.sbatch", tmp_path, "fast_left.txt", listed=contexts,
                               list_name="fast_left.txt", files=[(split, "lidar_box", name) for split, name in labeled],
                               fails=f"camera_image/{PRIVATE['broken']}")
    root = tmp_path / "home" / "root"
    assert calls == [f"storage cp --no-user-output-enabled gs://waymo_open_dataset_v_2_0_1/{split}/{component}/"
                     f"{name}.parquet {root}/{split}/{component}/"
                     for split, name in labeled for component in CAMERA_COMPONENTS]
    assert result.returncode == 0 and result.stderr == ""
    *lines, size = result.stdout.splitlines()
    assert lines == ["line 1: exit 0", "line 2: exit 2", "line 3: exit 1", "line 4: exit 0"]
    assert re.fullmatch(r"[\d.]+\w?", size.strip())
    assert not any(name in result.stdout + result.stderr for name in contexts)


def test_download_labels_fetches_calibration_with_the_labels_and_never_prints_a_name(tmp_path):
    """The scan counts FRONT_LEFT's boxes through each segment's own calibration, so it comes with the labels.
    gcloud names each object it copies, and rsync exits non-zero on the bucket's _metadata entry anyway. Each
    component gets its own number of files, so each count shows it's that directory's, and neither the _metadata
    entry nor a file gcloud is still writing counts."""
    names = [PRIVATE[name] for name in ("near", "far", "quiet", "busy")]
    result, _, calls = run_job("download_labels.sbatch", tmp_path, fails="storage rsync", objects=names)
    root = tmp_path / "home" / "root"
    splits = ("training", "validation")
    assert calls == [f"storage rsync --no-user-output-enabled --recursive gs://waymo_open_dataset_v_2_0_1/{split}/"
                     f"{component} {root}/{split}/{component}" for split in splits for component in LABEL_COMPONENTS]
    assert result.returncode == 0 and result.stderr == ""
    *lines, size = result.stdout.splitlines()
    copied = dict(zip(LABEL_COMPONENTS, (4, 3, 2, 1)))
    assert lines == [f"{split} {component} {copied[component]}" for split in splits for component in LABEL_COMPONENTS]
    assert re.fullmatch(r"[\d.]+\w?", size.strip())
    assert not any(name in result.stdout + result.stderr for name in names)


@pytest.mark.parametrize("job", ["download_listed.sbatch", "review_cutins.sbatch"])
@pytest.mark.parametrize("args", [[], ["cutin_top"], ["../selected/cutin_top.txt"]], ids=["none", "stem", "path"])
def test_a_job_takes_only_one_of_the_scans_two_lists(tmp_path, job, args):
    result, uv_calls, gcloud_calls = run_job(job, tmp_path, *args, listed=[PRIVATE["near"]],
                                             files=[("training", "lidar_box", PRIVATE["near"]),
                                                    ("training", "camera_image", PRIVATE["near"])])
    assert result.returncode == 2 and result.stdout == "" and not uv_calls and not gcloud_calls
    assert PRIVATE["near"] not in result.stderr


@pytest.mark.parametrize("job", ["download_labels.sbatch", "scan.sbatch", "download_listed.sbatch",
                                 "review_cutins.sbatch"])
def test_each_job_gives_the_order_to_run_them_in(job):
    assert ("# Run order: download_labels.sbatch, then scan.sbatch, then download_listed.sbatch for each list\n"
            "# (cutin_top.txt, fast_left.txt), then review_cutins.sbatch for each list.\n"
            ) in (REPO / "infra" / "pace" / job).read_text()


def test_the_scan_job_asks_for_what_the_spec_does(tmp_path):
    result, calls, _ = run_job("scan.sbatch", tmp_path)
    assert result.returncode == 0
    assert calls == ["run --frozen agatha-scan --config configs/stage1.toml --scan-config configs/scan.toml "
                     "--workers 8 --top 10"]
    header = (REPO / "infra" / "pace" / "scan.sbatch").read_text()
    assert "--cpus-per-task=8 --mem=32G" in header and "#SBATCH -t 1:00:00" in header


@pytest.mark.parametrize("job", ["download_listed.sbatch", "review_cutins.sbatch"])
def test_the_list_jobs_have_time_for_both_parts_of_the_list(job):
    """cutin_top.txt holds up to twice --top segments, 20 at the scan job's --top 10. Each job had 2 hours for 10."""
    assert "#SBATCH -t 4:00:00" in (REPO / "infra" / "pace" / job).read_text()


def test_both_list_jobs_take_twenty_segments_by_line_and_rank(tmp_path):
    """The longest list the scan job writes: 10 ranked, then 10 more that rank there with the steering test off."""
    names = [f"{str(i % 10) * 19}{i // 10}_100_000_120_000" for i in range(20)]
    result, _, calls = run_job("download_listed.sbatch", tmp_path / "download", "cutin_top.txt", listed=names,
                               files=[("training", "lidar_box", name) for name in names])
    assert result.returncode == 0 and len(calls) == 4 * len(names)
    assert result.stdout.splitlines()[:-1] == [f"line {line}: exit 0" for line in range(1, 21)]
    printed = result.stdout + result.stderr
    result, calls, _ = run_job("review_cutins.sbatch", tmp_path / "review", "cutin_top.txt", listed=names,
                               files=[("training", "camera_image", name) for name in names])
    assert calls == [*(STAGE1.format("training", name) for name in names), REVIEW]
    assert result.stdout.splitlines() == [*(f"cutin_{rank}: agatha-stage1 exit 0" for rank in range(1, 21)),
                                          f"uv {REVIEW}"]
    assert not any(name in printed + result.stdout + result.stderr for name in names)


def test_the_list_jobs_take_any_list_and_name_its_scenes_by_its_stem(tmp_path):
    """random15.txt, which pick_random.py writes: random15_<rank>, and no scan review pages, which are cutin_top's."""
    contexts = [PRIVATE["near"], PRIVATE["far"], PRIVATE["quiet"]]
    result, _, calls = run_job("download_listed.sbatch", tmp_path / "download", "random15.txt", listed=contexts,
                               list_name="random15.txt", files=[("training", "lidar_box", PRIVATE["near"]),
                                                                ("validation", "lidar_box", PRIVATE["quiet"])])
    assert result.returncode == 0 and len(calls) == 2 * len(CAMERA_COMPONENTS)
    assert result.stdout.splitlines()[:-1] == ["line 1: exit 0", "line 2: exit 1", "line 3: exit 0"]
    printed = result.stdout + result.stderr
    result, calls, _ = run_job("review_cutins.sbatch", tmp_path / "review", "random15.txt", listed=contexts,
                               list_name="random15.txt", files=[("training", "camera_image", PRIVATE["near"]),
                                                                ("validation", "camera_image", PRIVATE["quiet"])])
    assert result.returncode == 0
    assert calls == [STAGE1.format("training", PRIVATE["near"]), STAGE1.format("validation", PRIVATE["quiet"])]
    assert result.stdout.splitlines() == ["random15_1: agatha-stage1 exit 0", "random15_3: agatha-stage1 exit 1"]
    assert not any(name in printed + result.stdout + result.stderr for name in contexts)


@pytest.mark.parametrize("job", ["download_listed.sbatch", "review_cutins.sbatch"])
@pytest.mark.parametrize("name", ["Random15.txt", "random-15.txt", "random15.TXT", "random15.txt.bak", "sub/r.txt",
                                  ".txt"])
def test_a_list_job_takes_only_a_lists_plain_name(tmp_path, job, name):
    """[a-z0-9_]+.txt, as the list was written: never a path, and nothing that could pass for one."""
    result, uv_calls, gcloud_calls = run_job(job, tmp_path, name, listed=[PRIVATE["near"]], list_name="r.txt",
                                             files=[("training", "lidar_box", PRIVATE["near"]),
                                                    ("training", "camera_image", PRIVATE["near"])])
    assert result.returncode == 2 and result.stdout == "" and not uv_calls and not gcloud_calls
    assert "usage" in result.stderr and PRIVATE["near"] not in result.stderr


@pytest.mark.parametrize("job", ["download_listed.sbatch", "review_cutins.sbatch"])
def test_a_list_job_says_how_to_name_a_list(job):
    text = (REPO / "infra" / "pace" / job).read_text()
    assert f"# usage: {job} <list>.txt\n" in text and "random15.txt" in text and "[a-z0-9_]" in text

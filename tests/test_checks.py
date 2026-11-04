import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agatha.checks import (
    box_residuals, check_ego_velocity, check_gaps, check_joins, check_residuals, check_tracks, check_velocity,
    null_summary, output_hashes, pick_review_frames, timing_summary,
)
from agatha.columns import CAMERA_IDS, CAMERA_IMAGE, LIDAR_BOX, PROJECTED_BOX, SYNCED_BOX, VEHICLE_POSE
from agatha.config import load_config
from agatha.geometry import (
    CameraModel, box_corners, box_rectangle, inv_se3, rot_z, se3, transform_from_list, transform_points,
)
from agatha.loader import (
    Poses, load_cameras, load_image_metadata, load_lidar_boxes, load_poses, load_projected_boxes, load_synced_boxes,
)
from agatha.render import boxes_for_image
from agatha.report import Report
from agatha.scene_window import build_scene_window, write_scene_window
from agatha.store import SegmentOutputs, build_frames_table, build_segment_meta, load_segment, write_segment
from agatha.synthetic import OBJECT_IDS
from agatha.tracks import build_ego_table, build_tracks_table

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"
VERSION = "0.1.0+nogit.0123456789ab"
FRONT, FRONT_LEFT, FRONT_RIGHT = CAMERA_IDS["FRONT"], CAMERA_IDS["FRONT_LEFT"], CAMERA_IDS["FRONT_RIGHT"]
CAMERAS = (FRONT, FRONT_LEFT, FRONT_RIGHT)
TIME, OBJECT = "frame_timestamp_micros", "laser_object_id"
EDGES = ["e_u_min", "e_u_max", "e_v_min", "e_v_max"]
TILT = np.radians(0.3)  # about 11 px at FRONT's focal length
PITCH = np.array([[np.cos(TILT), 0.0, np.sin(TILT)], [0.0, 1.0, 0.0], [-np.sin(TILT), 0.0, np.cos(TILT)]])

# (argument of check_joins, component tag) for each keyed table.
TABLES = [("boxes", LIDAR_BOX), ("image_meta", CAMERA_IMAGE), ("synced", SYNCED_BOX), ("projected", PROJECTED_BOX)]

PINHOLE = CameraModel((1000.0, 1000.0, 960.0, 640.0, 0, 0, 0, 0, 0), np.eye(4), 1920, 1280)
SIZE = (4.0, 2.0, 1.5)
AHEAD = (20.0, 0.0, 1.0)
SIDEWAYS = (20.0, 1.0, 1.0)  # 1 m left of AHEAD, which is 50 px through PINHOLE
NO_SYNCED = (np.nan, np.nan, np.nan)


def process(root, ref, config, directory) -> SimpleNamespace:
    """What a run does before its checks: load the components, then build and write the tables."""
    poses = load_poses(root, ref)
    cameras = load_cameras(root, ref, CAMERAS)
    image_meta = load_image_metadata(root, ref, CAMERAS)
    boxes = load_lidar_boxes(root, ref)
    synced = load_synced_boxes(root, ref)
    projected = load_projected_boxes(root, ref, CAMERAS)
    ego, path = build_ego_table(poses, config.motion)
    tracks = build_tracks_table(boxes, synced, projected, poses, ego, path, config.motion, config.rear_axle_to_front_m)
    frames = build_frames_table(image_meta, poses, config.checks.gap_ms)
    write_segment(directory, build_segment_meta(ref, poses, cameras, config), frames, ego, tracks)
    segment = load_segment(directory)
    return SimpleNamespace(config=config, poses=poses, cameras=cameras, image_meta=image_meta, boxes=boxes,
                           synced=synced, projected=projected, segment=segment,
                           residuals=box_residuals(segment, projected, cameras, config))


@pytest.fixture(scope="module")
def run(synthetic_root, tmp_path_factory):
    """Every input the checks take. Shared, so tests copy before they corrupt anything."""
    root, ref = synthetic_root
    config = load_config(CONFIG_PATH, root=root, outputs=tmp_path_factory.mktemp("outputs"))
    return process(root, ref, config, config.outputs / ref.context_name)


def rows_of(table: pd.DataFrame, name: str) -> pd.DataFrame:
    return table[table[OBJECT] == OBJECT_IDS[name]]


def in_camera(table: pd.DataFrame, camera: int) -> pd.DataFrame:
    return table[table["camera"] == camera]


def shuffled(table: pd.DataFrame) -> pd.DataFrame:
    return table.sample(frac=1.0, random_state=0, ignore_index=True)


def with_limits(config, **limits):
    return replace(config, checks=replace(config.checks, **limits))


def with_tracks(run, tracks: pd.DataFrame) -> SegmentOutputs:
    return replace(run.segment, tracks=tracks)


def joins(run, **tables) -> dict:
    """check_joins on the run's tables, with some replaced."""
    every_table = {name: getattr(run, name) for name, _ in TABLES}
    return check_joins(run.poses, **{**every_table, **tables}, camera_ids=CAMERAS)


def with_jump(run, meters: float = 20.0) -> SegmentOutputs:
    """One position thrown sideways: an ID swap looks like this."""
    tracks = run.segment.tracks.copy()
    tracks.loc[rows_of(tracks, "lead").index[50], "pos_world_x_m"] += meters
    return with_tracks(run, tracks)


def with_camera_velocity(run, y: float = 0.0, cameras=(FRONT,)) -> pd.DataFrame:
    """image_meta with `y` m/s added to linear_velocity on these cameras."""
    added = run.image_meta["linear_velocity_y_mps"] + y * run.image_meta["camera"].isin(cameras)
    return run.image_meta.assign(linear_velocity_y_mps=added)


def labels_in_vehicle_axes(run, ego_relative: bool = False) -> SegmentOutputs:
    """Labels as they'd look if Waymo's `speed` weren't in world axes, under each wrong reading."""
    tracks = run.segment.tracks.copy()
    ego = run.segment.ego.set_index(TIME).loc[tracks[TIME]]
    x, y = tracks["v_label_world_x_mps"].to_numpy(), tracks["v_label_world_y_mps"].to_numpy()
    if ego_relative:
        x, y = x - ego["v_world_x_mps"].to_numpy(), y - ego["v_world_y_mps"].to_numpy()
    cos, sin = np.cos(ego["yaw_rad"].to_numpy()), np.sin(ego["yaw_rad"].to_numpy())
    tracks["v_label_world_x_mps"], tracks["v_label_world_y_mps"] = cos * x + sin * y, -sin * x + cos * y
    return with_tracks(run, tracks)


def corrupted(cam: CameraModel, rotation: np.ndarray) -> CameraModel:
    return CameraModel(cam.intrinsic, se3(rotation, np.zeros(3)) @ cam.extrinsic, cam.width, cam.height,
                       cam.rolling_shutter_direction)


def residual_rows(errors, timestamps=None, camera=FRONT, speed=10.0, yaw_rate=0.0, object_speed=10.0, column=960.0,
                  width=1920, kind="drawn") -> pd.DataFrame:
    """One row per error, which all four edges share. Any other column can be one value or one per row."""
    errors = np.asarray(errors, dtype=np.float64)
    return pd.DataFrame({
        TIME: np.arange(len(errors)) if timestamps is None else timestamps,
        "camera": camera,
        "ego_speed_mps": speed,
        "ego_yaw_rate_radps": yaw_rate,
        "object_speed_mps": object_speed,
        "column_px": column,
        "image_width_px": width,
        **{edge: errors for edge in EDGES},
        "kind": kind,
    })


def label_rows(errors, **columns) -> pd.DataFrame:
    """lidar_box rows beside one clean drawn box, so the status is about the label and isn't no_data."""
    return pd.concat([residual_rows([0.0]), residual_rows(errors, kind="lidar_box", **columns)])


def true_rectangle(center=AHEAD, pose=np.eye(4)) -> tuple[float, float, float, float]:
    """Where a box lands through PINHOLE, whose extrinsic is the identity."""
    return box_rectangle(transform_points(inv_se3(pose), box_corners(center, SIZE, 0.0)), PINHOLE)


def clamped(rectangle) -> tuple[float, float, float, float]:
    u_min, u_max, v_min, v_max = rectangle
    return max(u_min, 0.0), min(u_max, 1920.0), max(v_min, 0.0), min(v_max, 1280.0)


def one_box(center=AHEAD, oracle=None, image_pose=np.eye(4), synced_camera=-1, synced_center=NO_SYNCED,
            box_type="vehicle") -> tuple[SegmentOutputs, pd.DataFrame]:
    """A segment of one box in one frame, and its oracle, which defaults to where the box really lands."""
    u_min, u_max, v_min, v_max = true_rectangle(center, image_pose) if oracle is None else oracle
    key = {TIME: [1], OBJECT: ["ONE-BOX"]}
    tracks = pd.DataFrame({
        **key,
        "alias": ["V001"],
        "type": [box_type],
        "not_directly_observed": [False],
        **{f"center_{axis}_m": [value] for axis, value in zip("xyz", center)},
        **{f"size_{axis}_m": [value] for axis, value in zip("xyz", SIZE)},
        "heading_rad": [0.0],
        **{f"synced_center_{axis}_m": [value] for axis, value in zip("xyz", synced_center)},
        **{f"synced_size_{axis}_m": [value] for axis, value in zip("xyz", SIZE)},
        "synced_heading_rad": [0.0],
        "synced_camera": [synced_camera],
        "speed_mps": [3.0],
    })
    frames = pd.DataFrame({TIME: [1], "camera": [FRONT], "world_from_vehicle": [np.eye(4).ravel().tolist()],
                           "world_from_image_vehicle": [image_pose.ravel().tolist()]})
    ego = pd.DataFrame({TIME: [1], "speed_mps": [10.0], "yaw_rate_radps": [0.02]})
    projected = pd.DataFrame({**key, "camera": [FRONT], "center_x": [(u_min + u_max) / 2.0],
                              "center_y": [(v_min + v_max) / 2.0], "size_x": [u_max - u_min],
                              "size_y": [v_max - v_min]})
    return SegmentOutputs(Path("."), {}, frames, ego, tracks), projected


def scene(config, **box) -> pd.DataFrame:
    """box_residuals of one box in one frame."""
    return box_residuals(*one_box(**box), {FRONT: PINHOLE}, config)


def rendered(segment: SegmentOutputs) -> list[str]:
    """How render draws one_box's segment: the source of each box it shows."""
    return [box.draw_source for box in boxes_for_image(segment, 1, FRONT, PINHOLE)]


def none_for_nan(value):
    if isinstance(value, dict):
        return {key: none_for_nan(item) for key, item in value.items()}
    return None if isinstance(value, float) and np.isnan(value) else value


PASSING = {
    "joins": joins,
    "gaps": lambda run: check_gaps(run.poses, run.image_meta, run.config),
    "residuals": lambda run: check_residuals(run.residuals, run.config),
    "tracks": lambda run: check_tracks(run.segment, run.config),
    "velocity": lambda run: check_velocity(run.segment),
    "ego_velocity": lambda run: check_ego_velocity(run.segment, run.image_meta),
    "timing": lambda run: timing_summary(run.segment.frames, run.cameras),
    "nulls": lambda run: null_summary(run.boxes),
}
NO_DATA = {
    "joins": lambda run: joins(run, boxes=run.boxes[:0], synced=run.synced[:0], projected=run.projected[:0]),
    "gaps": lambda run: check_gaps(run.poses, run.image_meta[:0], run.config),
    "residuals": lambda run: check_residuals(run.residuals[:0], run.config),
    "tracks": lambda run: check_tracks(with_tracks(run, run.segment.tracks[:0]), run.config),
    "velocity": lambda run: check_velocity(with_tracks(run, run.segment.tracks[:0])),
    "ego_velocity": lambda run: check_ego_velocity(run.segment, run.image_meta[:0]),
    "timing": lambda run: timing_summary(run.segment.frames[:0], run.cameras),
    "nulls": lambda run: null_summary(run.boxes[:0]),
}
FAILING = {
    "joins": lambda run: joins(run, boxes=pd.concat([run.boxes, run.boxes[:1]])),
    "gaps": lambda run: check_gaps(run.poses, run.image_meta[1:], run.config),
    "residuals": lambda run: check_residuals(residual_rows([100.0] * 5), run.config),
    "tracks": lambda run: check_tracks(with_jump(run), run.config),
    "velocity": lambda run: check_velocity(labels_in_vehicle_axes(run)),
    "ego_velocity": lambda run: check_ego_velocity(run.segment, with_camera_velocity(run, y=5.0)),
}
STATUS_CASES = [pytest.param(check, status, id=f"{name}_{status}")
                for status, checks in (("pass", PASSING), ("no_data", NO_DATA), ("fail", FAILING))
                for name, check in checks.items()]


@pytest.mark.parametrize("check, status", STATUS_CASES)
def test_status_and_report_add(run, tmp_path, check, status):
    """Whatever the status, the report takes the dict as it is, and no row value is in it."""
    result = check(run)
    assert result["status"] == status

    report = Report("synthetic_cut_in", VERSION)
    report.add("check", **result)
    assert report.sections["check"] == none_for_nan(result)
    report.write(tmp_path / "report.json")  # refuses NaN

    text = json.dumps(report.sections)
    assert "synthetic_cut_in" not in text and not any(name in text for name in OBJECT_IDS.values())
    assert not re.search(r"(?<![\d.])\d{16}(?![\d.])", text)  # an epoch timestamp


def test_checks_print_and_log_nothing(run, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    box_residuals(run.segment, run.projected, run.cameras, run.config)
    pick_review_frames(run.residuals, run.segment)
    output_hashes(run.segment.directory)
    for check in (*PASSING.values(), *NO_DATA.values(), *FAILING.values()):
        check(run)
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


@pytest.mark.parametrize("table, tag", TABLES)
def test_a_row_one_microsecond_from_any_pose_is_an_orphan(run, table, tag):
    rows = getattr(run, table)
    orphan = rows[:1].assign(**{TIME: rows[TIME].iloc[0] + 1})
    result = joins(run, **{table: pd.concat([rows, orphan], ignore_index=True)})
    assert result["status"] == "fail"
    assert result["rows_without_pose"][tag] == 1
    assert result["duplicate_keys"][tag] == 0


@pytest.mark.parametrize("table, tag", [("synced", SYNCED_BOX), ("projected", PROJECTED_BOX)])
def test_an_unknown_object_is_an_orphan(run, table, tag):
    rows = getattr(run, table).copy()
    rows.loc[0, OBJECT] = "NOT-A-LABELED-OBJECT-00"
    result = joins(run, **{table: rows})
    assert result["status"] == "fail"
    assert result["rows_without_box"][tag] == 1


def test_an_object_must_have_a_box_at_that_very_timestamp(run):
    """The object and the timestamp both exist elsewhere, so only the pair can catch this."""
    row = rows_of(run.synced, "lead").iloc[10]
    is_row = (run.boxes[TIME] == row[TIME]) & (run.boxes[OBJECT] == row[OBJECT])
    result = joins(run, boxes=run.boxes[~is_row])
    assert result["status"] == "fail"
    assert result["rows_without_box"][SYNCED_BOX] == 1
    assert result["rows_without_box"][PROJECTED_BOX] == (rows_of(run.projected, "lead")[TIME] == row[TIME]).sum()


@pytest.mark.parametrize("table, tag", TABLES)
def test_a_repeated_key_fails(run, table, tag):
    rows = getattr(run, table)
    result = joins(run, **{table: pd.concat([rows, rows[:1]], ignore_index=True)})
    assert result["status"] == "fail"
    assert result["duplicate_keys"][tag] == 1


def test_a_repeated_pose_fails(run):
    poses = Poses(np.append(run.poses.timestamps, run.poses.timestamps[-1]),
                  np.concatenate([run.poses.world_from_vehicle, run.poses.world_from_vehicle[-1:]]))
    result = check_joins(poses, run.boxes, run.synced, run.projected, run.image_meta, CAMERAS)
    assert result["status"] == "fail"
    assert result["duplicate_keys"][VEHICLE_POSE] == 1


@pytest.mark.parametrize("table, tag", TABLES)
def test_float_timestamps_fail_even_where_they_compare_equal(run, table, tag):
    result = joins(run, **{table: getattr(run, table).astype({TIME: np.float64})})
    assert result["status"] == "fail"
    assert result["non_int64_keys"] == 1
    assert result["rows_without_pose"][tag] == 0


def test_a_requested_camera_with_no_images_fails(run):
    """Whatever's left still joins, so only the request can show the camera is gone."""
    result = joins(run, image_meta=run.image_meta[run.image_meta["camera"] != FRONT_LEFT],
                   projected=run.projected[run.projected["camera"] != FRONT_LEFT])
    assert result["status"] == "fail"
    assert result["cameras_without_images"] == 1


def test_joins_ignore_row_order(run):
    assert joins(run, **{name: shuffled(getattr(run, name)) for name, _ in TABLES}) == joins(run)


def test_a_dropped_frame_is_reported_and_is_no_failure(run):
    result = check_gaps(run.poses, run.image_meta, run.config)
    assert result["status"] == "pass"
    assert result["n_frames"] == 199 and result["n_gaps"] == 2
    assert result["max_dt_ms"] > run.config.checks.gap_ms
    assert result["frames_missing"] == {"front": 0, "front_left": 0, "front_right": 0}


def test_what_counts_as_a_gap_comes_from_the_config(run):
    assert check_gaps(run.poses, run.image_meta, with_limits(run.config, gap_ms=250.0))["n_gaps"] == 0


def test_no_frame_count_is_assumed(run):
    poses = Poses(run.poses.timestamps[:150], run.poses.world_from_vehicle[:150])
    image_meta = run.image_meta[run.image_meta[TIME].isin(poses.timestamps)]
    result = check_gaps(poses, image_meta, run.config)
    assert result["status"] == "pass" and result["n_frames"] == 150


def test_a_camera_losing_a_frame_alone_fails(run):
    image_meta = run.image_meta.drop(in_camera(run.image_meta, FRONT_RIGHT).index[5])
    result = check_gaps(run.poses, image_meta, run.config)
    assert result["status"] == "fail"
    assert result["frames_missing"] == {"front": 0, "front_left": 0, "front_right": 1}


def test_frames_missing_compares_timestamps_not_row_counts(run):
    image_meta = run.image_meta.copy()
    image_meta.loc[in_camera(image_meta, FRONT).index[5], TIME] += 1
    assert check_gaps(run.poses, image_meta, run.config)["frames_missing"]["front"] == 1


def test_a_single_frame_has_no_gaps_to_report(run):
    poses = Poses(run.poses.timestamps[:1], run.poses.world_from_vehicle[:1])
    assert check_gaps(poses, run.image_meta, run.config)["status"] == "no_data"


def test_residuals_are_near_zero_on_synthetic_data(run):
    """This proves the wiring only: the synthetic oracle comes from the same projection.

    Drawn rows sit a little further off, the reverse of real data. A synced box moves to its
    capture time and this oracle doesn't, so a moving object reads up to about 3 px.
    """
    assert list(run.residuals.columns) == [TIME, "camera", "ego_speed_mps", "ego_yaw_rate_radps", "object_speed_mps",
                                           "column_px", "image_width_px", *EDGES, "kind"]
    # No label_fallback: a box that fits inside one image here is always in its synced camera.
    assert set(run.residuals["kind"]) == {"lidar_box", "camera_synced_box", "frame_pose", "drawn"}
    lidar = run.residuals[run.residuals["kind"] == "lidar_box"]
    assert len(lidar) > 500
    assert np.abs(lidar[EDGES].to_numpy()).max() < 1e-6
    assert np.abs(run.residuals.loc[run.residuals["kind"] == "drawn", EDGES].to_numpy()).max() < 3.0


def test_residuals_carry_the_ego_speed_at_that_frame(run):
    speeds = run.segment.ego.set_index(TIME)["speed_mps"]
    np.testing.assert_array_equal(run.residuals["ego_speed_mps"], speeds.loc[run.residuals[TIME]])


def test_residuals_carry_the_ego_yaw_rate_and_the_objects_speed(run):
    rows = run.residuals[run.residuals["kind"] == "lidar_box"]
    yaw_rates = run.segment.ego.set_index(TIME)["yaw_rate_radps"]
    np.testing.assert_array_equal(rows["ego_yaw_rate_radps"], yaw_rates.loc[rows[TIME]])
    assert rows["object_speed_mps"].between(0.0, 20.0).all() and rows["object_speed_mps"].nunique() > 10


def test_edge_errors_keep_their_sign(run):
    """The rectangle is 10 px left of the oracle and 20 px below it: rectangle minus oracle, edge by edge."""
    u_min, u_max, v_min, v_max = true_rectangle()
    rows = scene(run.config, oracle=(u_min + 10.0, u_max + 10.0, v_min - 20.0, v_max - 20.0))
    np.testing.assert_allclose(rows.loc[rows["kind"] == "lidar_box", EDGES], [[-10.0, -10.0, 20.0, 20.0]])


def test_residuals_join_on_keys_not_row_order(run):
    segment = replace(run.segment, frames=shuffled(run.segment.frames), ego=shuffled(run.segment.ego),
                      tracks=shuffled(run.segment.tracks))
    rows = box_residuals(segment, shuffled(run.projected), run.cameras, run.config)
    keys = [TIME, "camera", "column_px", "kind", "e_u_min"]
    pd.testing.assert_frame_equal(rows.sort_values(keys, ignore_index=True),
                                  run.residuals.sort_values(keys, ignore_index=True))


@pytest.mark.parametrize("rotation", [rot_z(TILT) @ PITCH, rot_z(2.0 * TILT)], ids=["yaw_and_pitch", "yaw_alone"])
def test_a_slightly_wrong_extrinsic_fails(run, rotation):
    front = in_camera(run.projected, FRONT)
    good = box_residuals(run.segment, front, {FRONT: run.cameras[FRONT]}, run.config)
    assert check_residuals(good, run.config)["status"] == "pass"

    bad = box_residuals(run.segment, front, {FRONT: corrupted(run.cameras[FRONT], rotation)}, run.config)
    result = check_residuals(bad, run.config)
    assert result["status"] == "fail"
    assert result["drawn"]["median"] > run.config.checks.residual_median_px


def test_one_bad_camera_fails_however_few_boxes_it_has(run):
    """FRONT has over 50 boxes for each of FRONT_RIGHT's, so the pooled numbers can't show this."""
    limits = run.config.checks
    cameras = {**run.cameras, FRONT_RIGHT: corrupted(run.cameras[FRONT_RIGHT], rot_z(TILT) @ PITCH)}
    result = check_residuals(box_residuals(run.segment, run.projected, cameras, run.config), run.config)
    assert result["by_camera"]["front"]["boxes"] > 50 * result["by_camera"]["front_right"]["boxes"]
    assert result["drawn"]["median"] < limits.residual_median_px
    assert result["drawn"]["p95"] < limits.residual_p95_px
    assert result["by_camera"]["front_right"]["median"] > limits.residual_median_px
    assert result["by_camera"]["front"] == check_residuals(run.residuals, run.config)["by_camera"]["front"]
    assert result["status"] == "fail"


def test_a_configured_camera_with_no_boxes_has_shown_nothing(run):
    config = replace(run.config, cameras=CAMERAS)
    result = check_residuals(residual_rows([1.0], camera=FRONT), config)
    assert (result["status"], result["cameras_without_boxes"]) == ("no_data", 2)
    assert list(result["by_camera"]) == ["front"]


def test_a_camera_with_no_boxes_does_not_hide_a_failure(run):
    config = replace(run.config, cameras=CAMERAS)
    assert check_residuals(residual_rows([100.0], camera=FRONT), config)["status"] == "fail"  # a verdict needs no floor


def test_a_camera_is_counted_by_what_it_drew(run):
    """A label that fits inside the image isn't a box anyone saw, so these rows give FRONT_LEFT nothing to judge."""
    rows = pd.concat([residual_rows([1.0], camera=FRONT), residual_rows([1.0], camera=FRONT_LEFT, kind="lidar_box")])
    result = check_residuals(rows, replace(run.config, cameras=(FRONT, FRONT_LEFT)))
    assert (result["status"], result["cameras_without_boxes"]) == ("no_data", 1)


def test_a_camera_outside_the_config_counts_for_nothing(run):
    rows = pd.concat([residual_rows([1.0], camera=FRONT), residual_rows([1.0], camera=FRONT_LEFT)])
    result = check_residuals(rows, replace(run.config, cameras=(FRONT,)))
    assert (result["status"], result["cameras_without_boxes"]) == ("pass", 0)


def test_every_configured_camera_has_boxes_to_judge(run):
    result = check_residuals(run.residuals, replace(run.config, cameras=CAMERAS))
    assert (result["status"], result["cameras_without_boxes"]) == ("pass", 0)
    for camera in ("front", "front_left", "front_right"):
        assert result["by_camera"][camera]["boxes"] >= 10 and result["lidar_box_by_camera"][camera]["boxes"] >= 20


def through_the_frame_pose(run) -> SegmentOutputs:
    """Every image given the frame's pose, so the label path skips the step that makes it right."""
    frames = run.segment.frames.assign(world_from_image_vehicle=run.segment.frames["world_from_vehicle"])
    return replace(run.segment, frames=frames)


def label_a_meter_off(run) -> SegmentOutputs:
    """Every label 1 m to the left. A synced box has its own center, so nothing it draws moves."""
    tracks = run.segment.tracks
    return with_tracks(run, tracks.assign(center_y_m=tracks["center_y_m"] + 1.0))


def test_labels_through_the_frame_pose_show_in_the_label_numbers(run):
    """Each image has its own pose. FRONT_RIGHT is 14.5 ms from the labels, which is plenty at 12 m/s.

    This proves where the bug shows, not that the status fails. Every box this segment draws is
    synced, and a synced box uses no pose, so the limits see nothing. The label has fallen to
    exactly frame_pose's numbers, and that's the sign.
    """
    rows = box_residuals(through_the_frame_pose(run), run.projected, run.cameras, run.config)
    result = check_residuals(rows, run.config)
    assert result["status"] == "pass"
    label, frame_pose = result["lidar_box_by_camera"]["front_right"], result["frame_pose_by_camera"]["front_right"]
    assert label["median"] == pytest.approx(frame_pose["median"])
    assert label["median"] > run.config.checks.residual_median_px


def test_labels_through_the_frame_pose_fail_where_the_label_is_what_was_drawn(run):
    """The limits do judge this path, on the boxes that fall back to it. With no synced boxes, that's all of them."""
    segment = replace(through_the_frame_pose(run), tracks=run.segment.tracks.assign(synced_camera=-1))
    result = check_residuals(box_residuals(segment, run.projected, run.cameras, run.config), run.config)
    assert result["by_camera"]["front_right"]["median"] > run.config.checks.residual_median_px
    assert result["status"] == "fail"


def test_a_wrong_label_fails_on_front_however_good_the_drawn_boxes_are(run):
    """tracks.parquet holds these centers and no synced box reads them. This ego is too fast for the slow limits."""
    rows = box_residuals(label_a_meter_off(run), run.projected, run.cameras, run.config)
    result = check_residuals(rows, run.config)
    assert result["by_camera"] == check_residuals(run.residuals, run.config)["by_camera"]
    assert result["slow_and_steady"] == {"boxes": 0}
    assert result["status"] == "fail"


def test_a_wrong_label_fails_the_slow_limits_however_good_the_drawn_boxes_are(run):
    """The one limit on a side camera's label, and it's for a fault speed doesn't size, like this one.

    It can't be relied on for a pose fault: on an ego this slow the two poses are millimeters
    apart. The ego table is relabeled because this ego never drops under 6 m/s. FRONT_RIGHT
    alone, because FRONT's own limit would fail first.
    """
    ego = run.segment.ego.assign(speed_mps=0.5, yaw_rate_radps=0.0)
    config = replace(run.config, cameras=(FRONT_RIGHT,))
    front_right = in_camera(run.projected, FRONT_RIGHT)
    good = box_residuals(replace(run.segment, ego=ego), front_right, run.cameras, config)
    assert check_residuals(good, config)["status"] == "pass"

    bad = box_residuals(replace(label_a_meter_off(run), ego=ego), front_right, run.cameras, config)
    result = check_residuals(bad, config)
    assert result["status"] == "fail"
    assert result["by_camera"] == check_residuals(good, config)["by_camera"]
    assert result["slow_and_steady"]["median"] > config.checks.slow_residual_median_px


def test_the_image_pose_and_the_frame_pose_are_reported_side_by_side(run):
    """At p95, on a side camera, because FRONT's pose is 2 ms from the labels.

    This oracle has no object motion, so the image pose wins on every row here. On real data it
    wins only on stationary objects: traffic moving with the ego barely moves against the label's
    frame, so the frame pose reads it better, and the medians can go that way too.
    """
    result = check_residuals(run.residuals, run.config)
    for camera in ("front_left", "front_right"):
        assert result["frame_pose_by_camera"][camera]["p95"] > result["lidar_box_by_camera"][camera]["p95"] + 1.0
    assert result["frame_pose_by_camera"]["front"]["median"] < 1.0


def test_stationary_objects_show_where_the_image_pose_beats_the_frame_pose(run):
    """The comparison is read here: real traffic moving with the ego can reverse it on a camera's other rows."""
    stationary = check_residuals(run.residuals, run.config)["stationary_by_camera"]
    for camera in ("front_left", "front_right"):
        assert stationary[camera]["frame_pose"]["p95"] > stationary[camera]["lidar_box"]["p95"] + 1.0


def test_the_side_by_side_shows_poses_used_the_wrong_way_round(run):
    """Swapped poses double the label's error instead of removing it, so the frame pose would beat them.

    This proves the comparison can show a reversed chain. Nothing else can here: frame_pose and
    the drawn synced boxes read neither pose, and FRONT's pose is too near the labels to feel
    it, so the status stays pass.
    """
    frames = run.segment.frames.assign(world_from_vehicle=run.segment.frames["world_from_image_vehicle"],
                                       world_from_image_vehicle=run.segment.frames["world_from_vehicle"])
    rows = box_residuals(replace(run.segment, frames=frames), run.projected, run.cameras, run.config)
    result = check_residuals(rows, run.config)
    assert result["status"] == "pass"
    label, frame_pose = result["lidar_box_by_camera"]["front_right"], result["frame_pose_by_camera"]["front_right"]
    assert label["median"] > 1.5 * frame_pose["median"]


def wrong_frame_for_synced_boxes(run) -> SegmentOutputs:
    """Synced centers re-expressed in vehicle_pose's frame, a reading the spec infers against and nothing has tested."""
    tracks = run.segment.tracks.copy()
    frames = run.segment.frames.set_index([TIME, "camera"])["world_from_image_vehicle"]
    poses = run.segment.ego.set_index(TIME)["world_from_vehicle"]
    columns = ["synced_center_x_m", "synced_center_y_m", "synced_center_z_m"]
    for row in tracks[tracks["synced_camera"] >= 0].itertuples():
        image_pose = transform_from_list(frames[row.frame_timestamp_micros, row.synced_camera])
        moved = inv_se3(transform_from_list(poses[row.frame_timestamp_micros])) @ image_pose
        tracks.loc[row.Index, columns] = transform_points(moved, [getattr(row, name) for name in columns])[0]
    return with_tracks(run, tracks)


def test_a_synced_box_that_trails_the_label_on_stationary_objects_fails(run):
    """The path that draws most overlays. FRONT_LEFT's boxes stay inside the limits, so only this can show it."""
    limits = run.config.checks
    front_left = in_camera(run.projected, FRONT_LEFT)
    rows = box_residuals(wrong_frame_for_synced_boxes(run), front_left, run.cameras, run.config)
    result = check_residuals(rows, run.config)
    assert result["status"] == "fail"
    assert result["drawn"]["median"] < limits.residual_median_px and result["drawn"]["p95"] < limits.residual_p95_px
    stationary = result["stationary_by_camera"]["front_left"]
    assert stationary["camera_synced_box"]["p95"] > stationary["lidar_box"]["p95"] + 5.0


def test_stationary_objects_are_compared_per_camera_on_the_correct_frame(run):
    result = check_residuals(run.residuals, run.config)
    assert result["status"] == "pass"
    for camera in ("front_left", "front_right"):
        spreads = result["stationary_by_camera"][camera]
        assert spreads["camera_synced_box"]["boxes"] >= 5 and spreads["camera_synced_box"]["p95"] < 0.5


@pytest.mark.parametrize("object_speed, status", [(10.0, "pass"), (0.0, "fail")], ids=["moving", "stationary"])
def test_only_a_stationary_object_compares_the_synced_box_with_the_label(run, object_speed, status):
    """The drawn boxes are the same synced boxes, and stay inside the limits, so the comparison alone decides this.

    Only standing still takes object motion out of it, and that leaves the synced box's frame to blame.
    """
    errors = [0.0] * 15 + [6.0] * 5
    rows = pd.concat([residual_rows(errors, object_speed=object_speed, kind=kind)
                      for kind in ("drawn", "camera_synced_box")])
    rows = pd.concat([rows, residual_rows([0.0] * 20, object_speed=object_speed, kind="lidar_box")])
    assert check_residuals(rows, run.config)["status"] == status


def test_too_few_stationary_boxes_give_no_verdict_on_the_synced_box(run):
    rows = pd.concat([label_rows([0.0] * 10, object_speed=0.0),
                      residual_rows([50.0] * 4, object_speed=0.0, kind="camera_synced_box")])
    assert check_residuals(rows, run.config)["status"] == "pass"
    rows = pd.concat([rows, residual_rows([50.0], object_speed=0.0, kind="camera_synced_box")])
    assert check_residuals(rows, run.config)["status"] == "fail"


def test_a_box_in_the_open_has_rows_with_no_error(run):
    rows = scene(run.config)
    assert rows["kind"].tolist() == ["lidar_box", "frame_pose", "label_fallback", "drawn"]
    assert np.abs(rows[EDGES].to_numpy()).max() < 1e-9
    assert set(rows["column_px"]) == {960.0} and set(rows["image_width_px"]) == {1920}


@pytest.mark.parametrize("center", [(20.0, 18.0, 1.0), (20.0, -18.0, 1.0), (20.0, 0.0, 12.0), (20.0, 0.0, -12.0)],
                         ids=["left", "right", "top", "bottom"])
def test_a_clamped_rectangle_is_excluded(run, center):
    truth, oracle = true_rectangle(center), clamped(true_rectangle(center))
    assert np.abs(np.subtract(truth, oracle)).max() > run.config.checks.residual_p95_px  # kept, it would read as error
    assert scene(run.config, center=center, oracle=oracle).empty


@pytest.mark.parametrize("edge, on_margin, past_margin", [
    (0, 2.0, 1.5), (1, 1918.0, 1918.5), (2, 2.0, 1.5), (3, 1278.0, 1278.5),
], ids=["left", "right", "top", "bottom"])
def test_the_border_margin_includes_its_own_edge(run, edge, on_margin, past_margin):
    for value, rows in ((on_margin, 4), (past_margin, 0)):
        oracle = list(true_rectangle())
        oracle[edge] = value
        assert len(scene(run.config, oracle=oracle)) == rows


def test_both_rectangles_must_sit_inside(run):
    off_the_left = (20.0, 18.0, 1.0)
    assert true_rectangle(off_the_left)[0] < 0.0
    assert scene(run.config, center=off_the_left, oracle=(100.0, 300.0, 300.0, 500.0)).empty


def test_a_box_with_a_corner_behind_the_camera_is_excluded(run):
    straddling = (0.5, 0.0, 1.0)
    assert box_rectangle(box_corners(straddling, SIZE, 0.0), PINHOLE) is None
    assert scene(run.config, center=straddling, oracle=(100.0, 300.0, 300.0, 500.0)).empty


def test_every_clamped_synthetic_rectangle_is_left_out(run):
    margin = run.config.checks.border_margin_px
    half_u, half_v = run.projected["size_x"] / 2.0, run.projected["size_y"] / 2.0
    touching = ((run.projected["center_x"] - half_u < margin) | (run.projected["center_x"] + half_u > 1920 - margin)
                | (run.projected["center_y"] - half_v < margin) | (run.projected["center_y"] + half_v > 1280 - margin))
    assert touching.any()
    projected = run.projected.assign(touching=touching)

    lidar = run.residuals[run.residuals["kind"] == "lidar_box"]
    matched = lidar.merge(projected, left_on=[TIME, "camera", "column_px"], right_on=[TIME, "camera", "center_x"])
    assert len(matched) == len(lidar)
    assert not matched["touching"].any()

    # With no margin, only a corner the lens model rejects keeps a box out. That's the other way
    # to fall short, and the synthetic oracle's rectangle then doesn't have to reach the border.
    no_margin = with_limits(run.config, border_margin_px=-1e9)
    every_corner = box_residuals(run.segment, run.projected, run.cameras, no_margin)
    every_corner = every_corner[every_corner["kind"] == "lidar_box"]
    both = every_corner.merge(projected, left_on=[TIME, "camera", "column_px"], right_on=[TIME, "camera", "center_x"])
    assert len(lidar) == (~both["touching"]).sum() < (~touching).sum()


def test_clamped_rectangles_would_show_as_error_if_kept(run):
    config = with_limits(run.config, border_margin_px=-1e9)
    kept = box_residuals(run.segment, run.projected, run.cameras, config)
    result = check_residuals(kept, config)
    assert result["lidar_box"]["boxes"] > (run.residuals["kind"] == "lidar_box").sum()
    assert result["lidar_box"]["max"] > 100.0


def test_a_synced_box_adds_a_row_only_in_its_own_camera(run):
    here = scene(run.config, synced_camera=FRONT, synced_center=AHEAD)
    assert here["kind"].tolist() == ["lidar_box", "frame_pose", "camera_synced_box", "drawn"]
    elsewhere = scene(run.config, synced_camera=FRONT_RIGHT, synced_center=AHEAD)
    assert elsewhere["kind"].tolist() == ["lidar_box", "frame_pose", "label_fallback", "drawn"]


def test_synced_rectangle_uses_the_extrinsic_alone(run):
    """The image is 2 m on from the label's frame. A synced center already allows for that."""
    moved = se3(np.eye(3), [2.0, 0.0, 0.0])
    rows = scene(run.config, center=AHEAD, image_pose=moved, synced_camera=FRONT, synced_center=(18.0, 0.0, 1.0))
    rows = rows[rows["kind"] != "frame_pose"]  # the one reading here that ignores the 2 m
    assert rows["kind"].tolist() == ["lidar_box", "camera_synced_box", "drawn"]
    assert np.abs(rows[EDGES].to_numpy()).max() < 1e-9


def test_a_synced_rectangle_can_qualify_when_the_label_does_not(run):
    """Each rectangle stands alone. The label here falls off the image, and the synced box doesn't."""
    off_the_left = (20.0, 18.0, 1.0)
    rows = scene(run.config, center=off_the_left, oracle=true_rectangle(), synced_camera=FRONT, synced_center=AHEAD)
    # With no image pose it falls off too, so no frame_pose row.
    assert rows["kind"].tolist() == ["camera_synced_box", "drawn"]


@pytest.mark.parametrize("synced_camera, synced_center, source", [
    (FRONT, SIDEWAYS, "camera_synced_box"),
    (FRONT_RIGHT, SIDEWAYS, "lidar_box"),
    (-1, NO_SYNCED, "lidar_box"),
    (FRONT, NO_SYNCED, "lidar_box"),
    (FRONT, (20.0, np.nan, 1.0), "lidar_box"),
], ids=["this_camera", "another_camera", "none", "nan_values", "one_nan_value"])
def test_drawn_takes_the_path_render_takes(run, synced_camera, synced_center, source):
    """The synced rectangle where this is the box's camera, otherwise the label, and only the label is a fallback."""
    segment, projected = one_box(synced_camera=synced_camera, synced_center=synced_center)
    rows = box_residuals(segment, projected, {FRONT: PINHOLE}, run.config)
    [box] = boxes_for_image(segment, 1, FRONT, PINHOLE)
    drawn, fallback = (rows.loc[rows["kind"] == kind, EDGES] for kind in ("drawn", "label_fallback"))
    assert box.draw_source == source
    errors = np.subtract(box_rectangle(box.corners_camera, PINHOLE), true_rectangle())
    np.testing.assert_allclose(drawn, [errors], atol=1e-9)
    np.testing.assert_array_equal(fallback, drawn if source == "lidar_box" else drawn[:0])


def test_a_synced_box_with_a_corner_behind_the_camera_does_not_fall_back(run):
    """Render shows it, so the label can't answer for it. Only a missing synced box falls back."""
    segment, projected = one_box(synced_camera=FRONT, synced_center=(0.5, 0.0, 1.0))
    assert rendered(segment) == ["camera_synced_box"]
    rows = box_residuals(segment, projected, {FRONT: PINHOLE}, run.config)
    assert rows["kind"].tolist() == ["lidar_box", "frame_pose"]


@pytest.mark.parametrize("box_type", ["sign", "unknown"])
@pytest.mark.parametrize("synced_camera, synced_center, kinds", [
    (FRONT, AHEAD, ["lidar_box", "frame_pose", "camera_synced_box"]),
    (-1, NO_SYNCED, ["lidar_box", "frame_pose"]),
], ids=["synced", "no_synced"])
def test_a_type_render_skips_is_measured_but_not_drawn(run, box_type, synced_camera, synced_center, kinds):
    """A sign is a good still object to compare readings on, but no image shows it, so drawn can't answer for it.

    The limits on the label and the stationary comparison still read it, on purpose.
    """
    segment, projected = one_box(box_type=box_type, synced_camera=synced_camera, synced_center=synced_center)
    assert rendered(segment) == []
    assert box_residuals(segment, projected, {FRONT: PINHOLE}, run.config)["kind"].tolist() == kinds


@pytest.mark.parametrize("no_synced", [False, True], ids=["synced", "no_synced"])
def test_every_drawn_row_is_a_box_render_draws(run, no_synced):
    """Image by image, on tables as a run builds them. Oracle plus error gives a row's own center column."""
    tracks = run.segment.tracks.assign(synced_camera=-1) if no_synced else run.segment.tracks
    segment = with_tracks(run, tracks)
    rows = box_residuals(segment, run.projected, run.cameras, run.config)
    drawn = rows[rows["kind"] == "drawn"]
    assert len(drawn) > 500
    centers = drawn["column_px"] + (drawn["e_u_min"] + drawn["e_u_max"]) / 2.0
    for (timestamp, camera), columns in centers.groupby([drawn[TIME], drawn["camera"]]):
        cam = run.cameras[camera]
        boxes = boxes_for_image(segment, timestamp, camera, cam)
        shown = [sum(rectangle[:2]) / 2.0 for rectangle in (box_rectangle(box.corners_camera, cam) for box in boxes)
                 if rectangle is not None]
        assert all(np.isclose(shown, column, rtol=0.0, atol=1e-6).any() for column in columns)


def test_edges_are_pooled_and_keep_their_signed_medians(run):
    rows = residual_rows([0.0] * 5).assign(e_u_min=-1.0, e_u_max=2.0, e_v_min=3.0, e_v_max=-4.0)
    assert check_residuals(rows, run.config)["drawn"] == {
        "boxes": 5, "min": 1.0, "median": 2.5, "p95": 4.0, "max": 4.0, "signed_u_median": 0.5, "signed_v_median": -0.5}


def test_a_spread_over_few_boxes_reports_only_its_count(run):
    """One box's min, median, p95, and max solve for its four edge errors, and rows stay on PACE."""
    rows = residual_rows([1.0] * 4).assign(e_u_min=1.25, e_u_max=3.5, e_v_min=7.75, e_v_max=40.0)
    for count in (1, 4):
        assert check_residuals(rows[:count], run.config)["drawn"] == {"boxes": count}
    assert check_residuals(rows[:1], run.config)["by_camera"] == {"front": {"boxes": 1}}


def test_a_verdict_uses_the_numbers_a_small_bin_keeps_back(run):
    result = check_residuals(residual_rows([100.0]), run.config)
    assert result["status"] == "fail" and result["drawn"] == {"boxes": 1}


def test_no_residual_number_of_a_small_bin_reaches_the_report(run):
    rows = residual_rows([1.0] * 3 + [2.0] * 20, column=[100.0] * 3 + [1000.0] * 20, kind="lidar_box")
    by_column = check_residuals(rows, run.config)["by_column"]
    assert by_column["column_1"] == {"boxes": 3} and by_column["column_2"] == {"boxes": 0}
    assert by_column["column_3"]["median"] == 2.0


@pytest.mark.parametrize("error, status", [(5.0, "pass"), (5.01, "fail")])
def test_median_limit(run, error, status):
    assert check_residuals(residual_rows([error] * 10), run.config)["status"] == status


@pytest.mark.parametrize("outliers, status", [(4, "pass"), (6, "fail")])
def test_p95_limit_catches_what_the_median_misses(run, outliers, status):
    rows = residual_rows([0.0] * (100 - outliers) + [40.0] * outliers)
    result = check_residuals(rows, run.config)
    assert result["status"] == status
    assert result["drawn"]["median"] == 0.0


def test_slow_frames_have_tighter_limits(run):
    limits = run.config.checks
    errors = [0.0] * 99 + [3.0]
    fine_at_speed = label_rows(errors, speed=[10.0] * 99 + [limits.slow_speed_mps])
    assert check_residuals(fine_at_speed, run.config)["status"] == "pass"

    too_much_when_slow = label_rows(errors, speed=[10.0] * 99 + [0.5], object_speed=0.0)
    result = check_residuals(too_much_when_slow, run.config)
    assert result["status"] == "fail"
    assert result["by_ego_speed"]["slow"]["boxes"] == 1
    assert result["lidar_box"]["median"] < limits.slow_residual_median_px


def test_the_slow_limits_are_on_the_label_and_not_on_what_was_drawn(run):
    """On a slow, steady ego the label should be exact. A drawn box answers to the overall limits, slow or not."""
    rows = pd.concat([residual_rows([3.0] * 5, speed=0.5), residual_rows([0.0] * 5, speed=0.5, kind="lidar_box")])
    result = check_residuals(rows, run.config)
    assert result["status"] == "pass"
    assert result["drawn"]["median"] > run.config.checks.slow_residual_median_px


@pytest.mark.parametrize("yaw_rate, status", [(0.15, "pass"), (-0.15, "pass"), (0.04, "fail"), (-0.04, "fail")])
def test_slow_limits_need_a_steady_ego(run, yaw_rate, status):
    """Yaw moves the image during readout at any range, so correct code can't meet 1 px at 0.15 rad/s."""
    rows = label_rows([4.0] * 5, speed=0.9, yaw_rate=yaw_rate, object_speed=0.0)
    assert check_residuals(rows, run.config)["status"] == status


def test_a_stopped_ego_beside_moving_traffic_is_not_held_to_the_slow_limits(run):
    """The oracle moves a moving object and the label doesn't, so this gap is real on correct code."""
    moving = label_rows([4.0] * 5, speed=0.3, object_speed=12.0)
    assert check_residuals(moving, run.config)["status"] == "pass"
    still = label_rows([4.0] * 5, speed=0.3, object_speed=0.0)
    assert check_residuals(still, run.config)["status"] == "fail"


def test_an_unknown_yaw_rate_is_not_held_to_the_slow_limits(run):
    assert check_residuals(label_rows([3.0], speed=0.5, yaw_rate=np.nan), run.config)["status"] == "pass"


def test_the_slow_bin_follows_the_config(run):
    rows = label_rows([3.0], speed=1.5, object_speed=0.0)
    assert check_residuals(rows, run.config)["status"] == "pass"
    assert check_residuals(rows, with_limits(run.config, slow_speed_mps=2.0))["status"] == "fail"


def test_speed_bins_hold_their_lower_edge(run):
    speeds = [0.0, 0.99, 1.0, 4.99, 5.0, 14.99, 15.0, 40.0]
    bins = check_residuals(residual_rows([1.0] * 8, speed=speeds, kind="lidar_box"), run.config)["by_ego_speed"]
    assert [spread["boxes"] for spread in bins.values()] == [2, 2, 2, 2]


def test_residuals_are_binned_by_object_speed(run):
    speeds = [0.0, 0.99, 1.0, 4.99, 5.0, 14.99, 15.0, 40.0]
    rows = residual_rows([1.0] * 8, object_speed=speeds, kind="lidar_box")
    assert [spread["boxes"] for spread in check_residuals(rows, run.config)["by_object_speed"].values()] == [2, 2, 2, 2]


def test_an_unknown_ego_speed_counts_overall_and_in_no_bin(run):
    result = check_residuals(residual_rows([1.0], speed=np.nan, kind="lidar_box"), run.config)
    assert result["lidar_box"]["boxes"] == 1
    assert sum(spread["boxes"] for spread in result["by_ego_speed"].values()) == 0


@pytest.mark.parametrize("width, columns", [
    (1920, [0.0, 383.9, 384.0, 767.9, 768.0, 1151.9, 1152.0, 1535.9, 1536.0, 1919.9]),
    (1000, [0.0, 199.9, 200.0, 399.9, 400.0, 599.9, 600.0, 799.9, 800.0, 999.9]),
])
def test_column_bins_are_fifths_of_the_image_width(run, width, columns):
    rows = residual_rows([1.0] * 10, column=columns, width=width, kind="lidar_box")
    bins = check_residuals(rows, run.config)["by_column"]
    assert [spread["boxes"] for spread in bins.values()] == [2, 2, 2, 2, 2]


def test_a_side_cameras_label_far_over_the_limit_passes_if_every_drawn_box_is_fine(run):
    """The p95s of the real 29 m/s segment. A side camera's label can't be exact at that speed, and few boxes use it."""
    p95s = ((FRONT, 0.12, 3.3), (FRONT_LEFT, 4.76, 68.0), (FRONT_RIGHT, 3.45, 80.0))
    rows = pd.concat([residual_rows([error] * 20, camera=camera, speed=29.0, kind=kind)
                      for camera, drawn, label in p95s for kind, error in (("drawn", drawn), ("lidar_box", label))])
    fallback = residual_rows([14.5] * 5, camera=FRONT, speed=29.0, kind="label_fallback")
    result = check_residuals(pd.concat([rows, fallback, fallback.assign(kind="drawn")]),
                             replace(run.config, cameras=CAMERAS))
    assert result["status"] == "pass"
    assert result["by_camera"]["front_left"]["p95"] == 4.76
    assert result["lidar_box_by_camera"]["front_left"]["p95"] == 68.0
    assert result["lidar_box_by_camera"]["front_right"]["p95"] == 80.0
    assert result["label_fallback_by_camera"]["front"]["p95"] == 14.5
    assert result["lidar_box"]["boxes"] == 60
    assert result["by_ego_speed"]["from_15"]["boxes"] == 60  # the label stays in the bins that show pose and timing


def test_synced_rows_set_the_limits_where_they_are_what_was_drawn(run):
    """On that segment synced boxes drew 218 of 236 boxes shown, so a wrong one must fail, however good the label is."""
    rows = pd.concat([residual_rows([1.0] * 5, kind="lidar_box"), residual_rows([500.0] * 5, kind="camera_synced_box"),
                      residual_rows([500.0] * 5)])
    result = check_residuals(rows, run.config)
    assert result["status"] == "fail"
    assert result["lidar_box"]["max"] == 1.0
    assert result["camera_synced_box"]["median"] == result["drawn"]["median"] == 500.0
    for kind in ("camera_synced_box", "lidar_box"):  # neither is a box drawn, so neither has anything to judge
        assert check_residuals(rows[rows["kind"] == kind], run.config)["status"] == "no_data"


@pytest.mark.parametrize("box, path", [
    ({"synced_camera": FRONT, "synced_center": SIDEWAYS}, "camera_synced_box"),
    ({"center": SIDEWAYS, "oracle": true_rectangle()}, "label_fallback"),
    ({"center": SIDEWAYS, "oracle": true_rectangle(), "synced_camera": FRONT}, "label_fallback"),
    ({"center": SIDEWAYS, "oracle": true_rectangle(), "synced_camera": FRONT_RIGHT, "synced_center": AHEAD},
     "label_fallback"),
], ids=["synced", "fallback_no_synced_box", "fallback_nan_values", "fallback_synced_in_another_camera"])
def test_a_drawn_box_over_the_limit_fails_whichever_path_drew_it(run, box, path):
    """50 px off, by the path that drew it. The other path, where there is one, is exact, so it can't be what fails."""
    rows = scene(run.config, **box)
    assert path in set(rows["kind"])
    result = check_residuals(rows, run.config)
    assert result["status"] == "fail" and result["drawn"] == {"boxes": 1}
    assert result["label_fallback_by_camera"] == ({"front": {"boxes": 1}} if path == "label_fallback" else {})


def test_tracks_on_the_synthetic_segment(run):
    result = check_tracks(run.segment, run.config)
    zero_points = run.boxes["num_lidar_points"] == 0
    assert result["tracks"] == len(OBJECT_IDS) and result["rows"] == len(run.boxes)
    assert result["length_frames"]["max"] == len(run.poses.timestamps)
    assert result["length_frames"]["min"] == len(rows_of(run.boxes, "pedestrian"))
    assert result["zero_point_rows"] == result["zero_top_point_rows"] == zero_points.sum() > 0
    assert set(run.boxes.loc[zero_points, OBJECT]) == {OBJECT_IDS["occluded"]}


def test_zero_point_rows_tell_every_lidar_from_the_top_one(run):
    tracks = run.segment.tracks.copy()
    side_lidar_only = rows_of(tracks, "lead").index[:3]
    tracks.loc[side_lidar_only, "num_top_lidar_points"] = 0
    tracks.loc[side_lidar_only, "not_directly_observed"] = True
    before = check_tracks(run.segment, run.config)
    after = check_tracks(with_tracks(run, tracks), run.config)
    assert after["zero_point_rows"] == before["zero_point_rows"]
    assert after["zero_top_point_rows"] == before["zero_top_point_rows"] + 3


def test_a_frame_dropped_from_the_segment_is_no_gap_in_a_track(run):
    """Every long track spans the two dropped frames, so a gap counted in time would show here."""
    assert check_gaps(run.poses, run.image_meta, run.config)["n_gaps"] == 2
    assert check_tracks(run.segment, run.config)["internal_gaps"]["count"] == 0


def without_rows(run, name: str, positions) -> SegmentOutputs:
    """The segment, less some rows of one object's track."""
    tracks = run.segment.tracks
    return with_tracks(run, tracks.drop(rows_of(tracks, name).index[positions]))


def test_dropping_a_tracks_frames_reports_the_gap(run):
    result = check_tracks(without_rows(run, "lead", slice(50, 55)), run.config)
    assert result["status"] == "pass"  # occlusion makes these in real labels
    assert result["internal_gaps"]["count"] == result["internal_gaps"]["tracks"] == 1
    assert result["internal_gaps"]["frames"]["max"] == 5.0


def test_gap_sizes_are_summarized_across_tracks(run):
    segment = without_rows(run, "lead", [*range(50, 52), *range(100, 107)])
    gaps = check_tracks(segment, run.config)["internal_gaps"]
    assert gaps["count"] == 2 and gaps["tracks"] == 1
    assert gaps["frames"]["min"] == 2.0 and gaps["frames"]["median"] == 4.5 and gaps["frames"]["max"] == 7.0

    tracks = run.segment.tracks
    two_tracks = tracks.drop([rows_of(tracks, "beside").index[10], rows_of(tracks, "lead").index[10]])
    gaps = check_tracks(with_tracks(run, two_tracks), run.config)["internal_gaps"]
    assert gaps["count"] == gaps["tracks"] == 2


def test_a_gap_beside_a_dropped_frame_counts_only_frames_that_exist(run):
    """The track misses one real frame, though the hole in time is three frames long."""
    before_the_drop = int(np.diff(run.poses.timestamps).argmax())
    lead = rows_of(run.segment.tracks, "lead")
    assert (lead[TIME].to_numpy() == run.poses.timestamps).all()
    gaps = check_tracks(without_rows(run, "lead", [before_the_drop]), run.config)["internal_gaps"]
    assert gaps["count"] == 1 and gaps["frames"]["max"] == 1.0


def test_rows_lost_at_the_ends_of_a_track_are_no_gap(run):
    segment = without_rows(run, "lead", [0, 1, -2, -1])
    assert check_tracks(segment, run.config)["internal_gaps"]["count"] == 0


def test_type_changes_are_counted_and_are_no_failure(run):
    tracks = run.segment.tracks.copy()
    tracks.loc[rows_of(tracks, "pedestrian").index[20:30], "type"] = "cyclist"  # real labels do this
    result = check_tracks(with_tracks(run, tracks), run.config)
    assert result["type_changes"] == {"count": 2, "tracks": 1}
    assert result["status"] == "pass"


def test_a_jump_no_object_could_make_fails(run):
    result = check_tracks(with_jump(run), run.config)
    assert result["status"] == "fail"
    assert result["speed_spikes"] == {"count": 2, "tracks": 1}  # there and back. A speed here would be a row value.


def test_implied_speed_uses_real_time_across_a_dropped_frame(run):
    """Nothing here tops 16 m/s, but a 0.2 s step divided by 0.1 s would read as 32."""
    assert check_tracks(run.segment, with_limits(run.config, speed_spike_mps=20.0))["speed_spikes"]["count"] == 0
    assert check_tracks(run.segment, with_limits(run.config, speed_spike_mps=5.0))["status"] == "fail"


def test_velocity_labels_are_in_world_axes(run):
    result = check_velocity(run.segment)
    assert result["median_residual_world_mps"] < 0.05
    assert result["median_residual_vehicle_mps"] > 5.0
    assert result["median_residual_ego_relative_mps"] > 5.0


def test_labels_in_vehicle_axes_fail(run):
    result = check_velocity(labels_in_vehicle_axes(run))
    assert result["status"] == "fail"
    assert result["median_residual_vehicle_mps"] < 0.05 < result["median_residual_world_mps"]


def test_ego_relative_labels_fail(run):
    result = check_velocity(labels_in_vehicle_axes(run, ego_relative=True))
    assert result["status"] == "fail"
    assert result["median_residual_ego_relative_mps"] < 0.05 < result["median_residual_world_mps"]


def velocity_scene(yaw, ego_velocity=(50.0, 0.0), label=(10.0, 0.0), differences=((10.0, 1.0),)) -> dict:
    """check_velocity on rows sharing one label and ego, one row per difference."""
    n = len(differences)
    x, y = np.transpose(differences)
    tracks = pd.DataFrame({TIME: np.arange(n), "v_label_world_x_mps": label[0], "v_label_world_y_mps": label[1],
                           "v_diff_world_x_mps": x, "v_diff_world_y_mps": y,
                           "v_world_x_mps": label[0], "v_world_y_mps": label[1]})
    ego = pd.DataFrame({TIME: np.arange(n), "yaw_rad": yaw, "v_world_x_mps": ego_velocity[0],
                        "v_world_y_mps": ego_velocity[1]})
    return check_velocity(SegmentOutputs(Path("."), {}, pd.DataFrame(), ego, tracks))


@pytest.mark.parametrize("yaw, ego_velocity, status", [
    (0.30, (50.0, 0.0), "pass"),    # 1.0 m/s against 2.0 and more
    (0.25, (50.0, 0.0), "fail"),    # 1.0 against 1.5: better, but not by half
    (0.30, (0.45, -1.96), "fail"),  # far from vehicle axes, but ego-relative fits
], ids=["clear_win", "narrow_win", "loses_to_ego_relative"])
def test_world_axes_must_be_under_half_of_both_others(yaw, ego_velocity, status):
    """A label of (10, 0) against a difference of (10, 1), so the world reading is always 1 m/s off."""
    result = velocity_scene(yaw, ego_velocity)
    assert result["median_residual_world_mps"] == 1.0
    assert result["status"] == status


@pytest.mark.parametrize("degrees, noise_mps", [(0.0, 0.14), (1.5, 0.14), (3.0, 0.3), (-3.0, 0.3)])
def test_an_ego_heading_near_due_east_gives_no_verdict(degrees, noise_mps):
    """Vehicle axes are world axes there, so correct labels can't beat them by half once there's any noise."""
    yaw = np.radians(degrees)
    label = 12.0 * np.cos(yaw), 12.0 * np.sin(yaw)
    differences = np.random.default_rng(0).normal(label, noise_mps, (200, 2))
    assert velocity_scene(yaw, label, label, differences) == {"status": "no_data", "rows": 0}


@pytest.mark.parametrize("degrees, noise_mps", [(5.0, 0.3), (30.0, 0.3), (180.0, 0.3)])
def test_the_same_traffic_off_east_passes(degrees, noise_mps):
    yaw = np.radians(degrees)
    label = 12.0 * np.cos(yaw), 12.0 * np.sin(yaw)
    differences = np.random.default_rng(0).normal(label, noise_mps, (200, 2))
    assert velocity_scene(yaw, label, label, differences)["status"] == "pass"


def test_only_rows_that_separate_the_readings_are_compared():
    """Row by row, not by segment: part of a turn can head east."""
    tracks = pd.DataFrame({TIME: [0, 1], "v_label_world_x_mps": [12.0, 12.0], "v_label_world_y_mps": [0.0, 0.0],
                           "v_diff_world_x_mps": [12.0, 12.0], "v_diff_world_y_mps": [0.1, 0.1],
                           "v_world_x_mps": [12.0, 12.0], "v_world_y_mps": [0.0, 0.0]})
    ego = pd.DataFrame({TIME: [0, 1], "yaw_rad": [0.0, 0.5], "v_world_x_mps": 10.0, "v_world_y_mps": 0.0})
    assert check_velocity(SegmentOutputs(Path("."), {}, pd.DataFrame(), ego, tracks))["rows"] == 1


def test_pose_differences_agree_with_the_cameras_reading_of_ego_velocity(run):
    result = check_ego_velocity(run.segment, run.image_meta)
    assert result["rows"] == len(run.poses.timestamps)
    assert result["residual_mps"]["median"] < 0.01
    assert result["residual_mps"]["max"] < 0.15  # the ends are one-sided: they read 0.1 s away, further round the curve


@pytest.mark.parametrize("y, status", [(0.9, "pass"), (1.1, "fail")])
def test_ego_velocity_fails_past_its_limit(run, y, status):
    assert check_ego_velocity(run.segment, with_camera_velocity(run, y))["status"] == status


def test_ego_velocity_in_the_wrong_axes_fails(run):
    """The ego's world y velocity is most of its speed here, so swapping it with x is a big error."""
    swapped = run.image_meta.rename(columns={"linear_velocity_x_mps": "linear_velocity_y_mps",
                                             "linear_velocity_y_mps": "linear_velocity_x_mps"})
    assert check_ego_velocity(run.segment, swapped)["status"] == "fail"


def test_ego_velocity_reads_front_alone(run):
    image_meta = with_camera_velocity(run, 100.0, cameras=(FRONT_LEFT, FRONT_RIGHT))
    assert check_ego_velocity(run.segment, image_meta)["residual_mps"]["max"] < 0.15


def test_ego_velocity_has_nothing_to_say_without_front(run):
    assert check_ego_velocity(run.segment, in_camera(run.image_meta, FRONT_LEFT))["status"] == "no_data"


def test_an_unknown_ego_velocity_is_left_out_of_the_comparison(run):
    ego = run.segment.ego.copy()
    ego.loc[3, "v_world_y_mps"] = np.nan
    result = check_ego_velocity(replace(run.segment, ego=ego), run.image_meta)
    assert result["rows"] == len(ego) - 1 and result["status"] == "pass"


def test_velocity_compares_only_moving_rows_with_a_label(run):
    tracks = run.segment.tracks
    moving = np.hypot(tracks["v_diff_world_x_mps"], tracks["v_diff_world_y_mps"]) > 1.0
    labeled = tracks["v_label_world_x_mps"].notna()
    assert labeled.sum() < len(tracks) and moving.sum() < len(tracks)  # both really do filter
    assert check_velocity(run.segment)["rows"] == (labeled & moving).sum()


def test_an_unknown_ego_velocity_does_not_spoil_the_medians(run):
    ego = run.segment.ego.copy()
    ego.loc[:50, "v_world_x_mps"] = np.nan
    result = check_velocity(replace(run.segment, ego=ego))
    assert result["status"] == "pass"
    assert 0 < result["rows"] < check_velocity(run.segment)["rows"]


def test_velocity_has_nothing_to_compare_without_labels_or_motion(run):
    tracks = run.segment.tracks
    no_labels = tracks.assign(v_label_world_x_mps=np.nan, v_label_world_y_mps=np.nan)
    no_motion = tracks.assign(v_diff_world_x_mps=0.0, v_diff_world_y_mps=0.0)
    for table in (no_labels, no_motion):
        assert check_velocity(with_tracks(run, table)) == {"status": "no_data", "rows": 0}


def test_timing_names_each_cameras_readout_direction(run):
    """TIME-3's 15 ms against 29 ms edge gap hangs on this, and only the report leaves PACE."""
    cameras = timing_summary(run.segment.frames, run.cameras)["cameras"]
    assert {camera["rolling_shutter_direction"] for camera in cameras.values()} == {"LEFT_TO_RIGHT"}


def test_timing_reads_each_cameras_offsets(run):
    cameras = timing_summary(run.segment.frames, run.cameras)["cameras"]
    offsets = {"front": 50.0, "front_left": 37.5, "front_right": 62.5}
    assert set(cameras) == set(offsets)
    for name, offset in offsets.items():
        assert cameras[name]["capture_offset_ms"]["median"] == pytest.approx(offset, abs=1e-3)
        assert cameras[name]["exposure_mid_after_pose_ms"]["median"] == pytest.approx(-4.3, abs=1e-3)
        assert cameras[name]["readout_ms"]["median"] == pytest.approx(44.0, abs=1e-3)
        assert 1.0 <= cameras[name]["shutter_ms"]["min"] <= cameras[name]["shutter_ms"]["max"] <= 10.0


def test_timing_definitions():
    frames = pd.DataFrame({
        "camera": [FRONT, FRONT, FRONT_RIGHT],
        "capture_offset_ms": [40.0, 60.0, 0.0],
        "trigger_time_s": [100.0, 100.0, 0.0],
        "readout_done_time_s": [100.05, 100.05, 0.0],
        "pose_timestamp_s": [100.02, 100.02, 0.0],
        "shutter_s": [0.002, 0.004, 0.0],
    })
    front = timing_summary(frames, {FRONT: PINHOLE})["cameras"]["front"]
    assert (front["capture_offset_ms"]["min"], front["capture_offset_ms"]["max"]) == (40.0, 60.0)
    assert front["capture_offset_ms"]["median"] == 50.0
    assert front["exposure_mid_after_pose_ms"]["median"] == pytest.approx(5.0, abs=1e-6)
    assert front["readout_ms"]["max"] == pytest.approx(48.0, abs=1e-6)
    assert front["readout_ms"]["min"] == pytest.approx(46.0, abs=1e-6)
    assert front["shutter_ms"]["median"] == pytest.approx(3.0)


def test_null_summary_shares():
    boxes = pd.DataFrame({
        "speed_x": [1.0, np.nan, 2.0, 3.0], "speed_y": [1.0, np.nan, 2.0, 3.0], "speed_z": [0.0, np.nan, 0.0, 0.5],
        "accel_x": [np.nan, np.nan, 1.0, 1.0], "accel_y": [np.nan, np.nan, 1.0, 1.0],
    })
    result = null_summary(boxes)
    assert result["rows"] == 4
    assert result["speed_null_share"] == 0.25 and result["accel_null_share"] == 0.5
    assert result["speed_z_zero_share"] == pytest.approx(2.0 / 3.0)  # of the three rows with a value


def test_null_summary_on_the_synthetic_segment(run):
    result = null_summary(run.boxes)
    assert 0.0 < result["speed_null_share"] < 0.1
    assert result["speed_z_zero_share"] == 1.0  # speed.z is always 0 there, as a filled-in value would be


def test_null_summary_when_optional_columns_were_absent(run):
    absent = run.boxes.assign(**{name: np.nan for name in run.boxes.columns if name.startswith(("speed_", "accel_"))})
    result = null_summary(absent)
    assert result["status"] == "pass"
    assert result["speed_null_share"] == result["accel_null_share"] == 1.0
    assert np.isnan(result["speed_z_zero_share"])


def picked(picks: list, reason: str) -> list[tuple[int, int]]:
    return [(timestamp, camera) for timestamp, camera, why in picks if why == reason]


def test_review_quotas(run):
    picks = pick_review_frames(run.residuals, run.segment)
    counts = Counter(reason for _, _, reason in picks)
    assert set(counts) == {"worst_residual", "image_edge", "near_fast", "random"}
    assert counts["random"] <= 12 and max(counts["worst_residual"], counts["image_edge"], counts["near_fast"]) <= 6


def test_review_frames_are_real_frames_with_exact_keys(run):
    picks = pick_review_frames(run.residuals, run.segment)
    frames = [pick[:2] for pick in picks]
    assert len(frames) == len(set(frames))
    assert set(frames) <= set(zip(run.segment.frames[TIME].tolist(), run.segment.frames["camera"].tolist()))
    assert all(type(timestamp) is int and type(camera) is int for timestamp, camera in frames)


def test_review_frames_repeat_and_the_seed_moves_them(run):
    first = pick_review_frames(run.residuals, run.segment, seed=1)
    assert first == pick_review_frames(run.residuals, run.segment, seed=1)
    assert first != pick_review_frames(run.residuals, run.segment, seed=2)


def test_worst_residuals_rank_frames_not_boxes(run):
    """Seven bad boxes in one frame take one place, so six different frames still come back."""
    times = run.segment.frames[TIME].unique()[:8].tolist()
    rows = residual_rows([50.0] * 7 + list(range(1, 8)), timestamps=[times[0]] * 7 + times[1:])
    worst = picked(pick_review_frames(rows, run.segment), "worst_residual")
    assert worst == [(time, FRONT) for time in (times[0], times[7], times[6], times[5], times[4], times[3])]


def test_worst_residuals_rank_by_size_whatever_the_sign(run):
    first, second = run.segment.frames[TIME].unique()[:2].tolist()
    rows = residual_rows([1.0, -9.0], timestamps=[first, second])
    assert picked(pick_review_frames(rows, run.segment), "worst_residual")[0] == (second, FRONT)


@pytest.mark.parametrize("kind", ["lidar_box", "frame_pose", "camera_synced_box", "label_fallback"])
def test_review_ranks_only_what_was_drawn(run, kind):
    """At speed the label's worst frames are rolling shutter, and no image shows them, so they'd waste the list."""
    first, second = run.segment.frames[TIME].unique()[:2].tolist()
    rows = pd.concat([residual_rows([1.0], timestamps=[first]),
                      residual_rows([500.0], timestamps=[second], column=5.0, kind=kind)])
    picks = pick_review_frames(rows, run.segment)
    assert picked(picks, "worst_residual") == [(first, FRONT)]
    assert picked(picks, "image_edge") == []  # the drawn frame keeps its first reason, and the other isn't drawn


def test_image_edge_frames_come_from_either_side(run):
    """Six frames with worse errors fill their own list, which leaves these two to show a reason."""
    times = run.segment.frames[TIME].unique()[:8].tolist()
    rows = pd.concat([residual_rows([10.0] * 6, timestamps=times[:6]),
                      residual_rows([0.0, 0.0], timestamps=times[6:], column=[1900.0, 30.0])])
    assert picked(pick_review_frames(rows, run.segment), "image_edge") == [(times[6], FRONT), (times[7], FRONT)]


def test_near_fast_needs_a_camera_that_has_the_object_in_frame(run):
    tracks = run.segment.tracks.copy()
    row = rows_of(tracks, "lead").index[10]
    timestamp = tracks.at[row, TIME].item()
    tracks.loc[row, ["center_x_m", "center_y_m", "speed_mps"]] = [2.0, 0.0, 200.0]

    tracks.at[row, "fov_cameras"] = [FRONT, FRONT_RIGHT]
    near_fast = picked(pick_review_frames(run.residuals[:0], with_tracks(run, tracks)), "near_fast")
    assert set(near_fast[:2]) == {(timestamp, FRONT), (timestamp, FRONT_RIGHT)}

    tracks.at[row, "fov_cameras"] = []
    near_fast = picked(pick_review_frames(run.residuals[:0], with_tracks(run, tracks)), "near_fast")
    assert timestamp not in {time for time, _ in near_fast}


def test_near_fast_never_names_an_image_that_is_not_there(run):
    """projected_lidar_box can have rows for a frame with no image, and the contact sheet would fail on it."""
    near_fast = picked(pick_review_frames(run.residuals, run.segment), "near_fast")
    hole = pd.MultiIndex.from_tuples(near_fast[:1], names=[TIME, "camera"])
    frames = run.segment.frames
    without = frames[~pd.MultiIndex.from_frame(frames[[TIME, "camera"]]).isin(hole)]
    picks = pick_review_frames(run.residuals, replace(run.segment, frames=without))
    assert near_fast[0] not in [pick[:2] for pick in picks]


def test_near_fast_weighs_range_against_speed(run):
    """A slower object close by outranks a faster one far off."""
    tracks = run.segment.tracks.copy()
    near, far = rows_of(tracks, "lead").index[[10, 20]]
    columns = ["center_x_m", "center_y_m", "speed_mps"]
    tracks.loc[near, columns] = [2.0, 0.0, 20.0]
    tracks.loc[far, columns] = [70.0, 0.0, 30.0]
    near_fast = picked(pick_review_frames(run.residuals[:0], with_tracks(run, tracks)), "near_fast")
    assert near_fast[0] == (tracks.at[near, TIME].item(), FRONT)


def test_a_frame_keeps_its_first_reason(run):
    """With five frames, random picks them all, including the two already picked as worst."""
    frames = run.segment.frames.iloc[:5]
    worst = frames.iloc[:2]
    rows = residual_rows([9.0, 8.0], timestamps=worst[TIME].tolist(), camera=worst["camera"].tolist())
    picks = pick_review_frames(rows, replace(run.segment, frames=frames))
    assert picked(picks, "worst_residual") == list(zip(worst[TIME].tolist(), worst["camera"].tolist()))
    assert len({pick[:2] for pick in picks}) == len(picks)
    assert {pick[:2] for pick in picks} >= set(zip(frames[TIME].tolist(), frames["camera"].tolist()))


def test_review_still_works_with_no_qualifying_boxes(run):
    picks = pick_review_frames(run.residuals[:0], run.segment)
    assert picked(picks, "worst_residual") == picked(picks, "image_edge") == []
    assert picked(picks, "random") and picked(picks, "near_fast")


def write_tree(directory: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_two_identical_runs_hash_the_same(synthetic_root, run, tmp_path):
    root, ref = synthetic_root
    for name, report in (("first", b"run one"), ("second", b"run two")):
        directory = tmp_path / name
        segment = process(root, ref, run.config, directory).segment
        window = build_scene_window(segment, run.config, 0.0, 5.0, 2.0, (FRONT,), "pass1")
        write_scene_window(window, directory)
        # Each run writes these differently, which is why they're left out.
        write_tree(directory, {"qa/contact_sheet.html": report, "clips/clip.mp4": report})

    hashes = output_hashes(tmp_path / "first")
    assert hashes == output_hashes(tmp_path / "second")
    window_name = next(iter((tmp_path / "first" / "windows").iterdir())).name
    assert set(hashes) == {"segment.json", "frames.parquet", "ego.parquet", "tracks.parquet", f"windows/{window_name}"}


@pytest.mark.parametrize("change", [
    lambda directory: (directory / "render" / "FRONT" / "1__clean.jpg").write_bytes(b"other pixels"),
    lambda directory: (directory / "windows" / "extra.json").write_bytes(b"{}"),
    lambda directory: (directory / "tracks.parquet").unlink(),
], ids=["changed", "added", "removed"])
def test_any_changed_file_changes_the_hashes(tmp_path, change):
    files = {"tracks.parquet": b"table", "windows/w.json": b"{}", "render/FRONT/1__clean.jpg": b"pixels"}
    write_tree(tmp_path, files)
    before = output_hashes(tmp_path)
    change(tmp_path)
    assert output_hashes(tmp_path) != before


def test_hashes_are_keyed_by_relative_path(tmp_path):
    write_tree(tmp_path, {"render/FRONT/1__clean.jpg": b"pixels"})
    assert output_hashes(tmp_path) == {"render/FRONT/1__clean.jpg": hashlib.sha256(b"pixels").hexdigest()}


def test_only_the_top_level_outputs_are_left_out(tmp_path):
    write_tree(tmp_path, {"qa/a.html": b"", "clips/a.mp4": b"", "render/qa/a.jpg": b"", "render/clips/a.jpg": b""})
    assert set(output_hashes(tmp_path)) == {"render/qa/a.jpg", "render/clips/a.jpg"}


def test_the_report_refuses_paths_and_hashes_but_takes_the_verdict(tmp_path):
    write_tree(tmp_path, {"tracks.parquet": b"table"})
    hashes = output_hashes(tmp_path)
    report = Report("synthetic_cut_in", VERSION)
    with pytest.raises(ValueError):
        report.add("repro", **hashes)
    report.add("repro", hashes_match=hashes == output_hashes(tmp_path))
    assert report.sections["repro"] == {"hashes_match": True}

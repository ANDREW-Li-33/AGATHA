import numpy as np
import pandas as pd
import pytest

from agatha.config import MotionConfig
from agatha.geometry import rot_z, se3
from agatha.loader import Poses
from agatha.tracks import EgoPath, assign_aliases, build_ego_table, build_tracks_table, central_difference

T0 = 1_550_000_000_000_000
MOTION = MotionConfig(stationary_mps=0.2, velocity_baseline_s=0.2, accel_baseline_s=0.5, max_baseline_s=1.0)
TIME, OBJECT = "frame_timestamp_micros", "laser_object_id"
STEP_S = 0.1


def times(n: int, dropped=()) -> np.ndarray:
    return np.delete(np.arange(n) * STEP_S, list(dropped))


def poses_at(t_s, x, y=0.0, yaw=0.0, z=0.0) -> Poses:
    """A Poses from arrays, any of which can be one value."""
    t_s = np.asarray(t_s, dtype=np.float64)
    x, y, yaw, z = (np.broadcast_to(values, t_s.shape) for values in (x, y, yaw, z))
    return Poses(T0 + np.round(t_s * 1e6).astype(np.int64),
                 np.stack([se3(rot_z(a), [i, j, k]) for i, j, a, k in zip(x, y, yaw, z)]))


def box_rows(poses: Poses, center, object_id="LEAD", size=(4.6, 1.9, 1.6), heading=0.0, speed=(np.nan, np.nan),
             accel=(np.nan, np.nan)) -> pd.DataFrame:
    """lidar_box rows, tidied as by the loader, for one object in every frame. `center` can be one per frame."""
    n = len(poses.timestamps)
    x, y, z = np.broadcast_to(center, (n, 3)).T
    return pd.DataFrame({
        TIME: poses.timestamps, OBJECT: object_id, "center_x": x, "center_y": y, "center_z": z,
        "size_x": size[0], "size_y": size[1], "size_z": size[2], "heading": heading, "type": 1,
        "num_lidar_points": 100, "num_top_lidar_points": 80, "speed_x": speed[0], "speed_y": speed[1], "speed_z": 0.0,
        "accel_x": accel[0], "accel_y": accel[1], "accel_z": 0.0,
    })


def tracks_of(poses: Poses, boxes: pd.DataFrame, motion=MOTION, front_m=4.0) -> pd.DataFrame:
    """build_tracks_table with no synced or projected rows, which nothing here is about."""
    ego, path = build_ego_table(poses, motion)
    synced = pd.DataFrame(columns=[TIME, OBJECT, "most_visible_camera", "center_x", "center_y", "center_z",
                                   "size_x", "size_y", "size_z", "heading"]).astype({TIME: np.int64})
    projected = pd.DataFrame(columns=[TIME, OBJECT, "camera"]).astype({TIME: np.int64})
    return build_tracks_table(boxes, synced, projected, poses, ego, path, motion, front_m)


def world_to_vehicle(poses: Poses, point) -> np.ndarray:
    """A fixed world point in each frame's vehicle frame, as a stationary object's label reads."""
    relative = np.append(point, 1.0)
    return np.stack([np.linalg.inv(pose) @ relative for pose in poses.world_from_vehicle])[:, :3]


# central_difference

def test_a_straight_line_has_its_slope_everywhere_including_the_ends():
    t = times(20)
    slope, _ = central_difference(3.0 * t + 1.0, t, 0.2, 1.0)
    np.testing.assert_allclose(slope, 3.0)


def test_the_window_is_symmetric_where_it_can_be_and_one_sided_at_the_ends():
    t = times(10)
    values = t * t
    slope, midpoints = central_difference(values, t, 0.2, 1.0)
    np.testing.assert_allclose(slope[1:-1], 2.0 * t[1:-1])  # a parabola's chord has the slope at its middle
    np.testing.assert_allclose(midpoints[[0, 1, 2, -2, -1]], t[[1, 1, 2, -2, -2]])
    np.testing.assert_allclose(slope[0], (values[2] - values[0]) / 0.2)


def test_a_baseline_widens_past_the_one_step_that_noise_dominates():
    """Noise that flips each frame cancels over two steps, and reads 2 m/s over one."""
    t = times(20)
    flipping = t + 0.1 * (-1.0) ** np.arange(20)
    slope, _ = central_difference(flipping, t, 0.2, 1.0)
    np.testing.assert_allclose(slope, 1.0)


def test_jitter_does_not_widen_a_window_that_nearly_spans_the_baseline():
    t = np.array([0.0, 0.0998, 0.1996, 0.2994, 0.3992])
    _, midpoints = central_difference(t, t, 0.2, 1.0)
    np.testing.assert_allclose(midpoints[2], t[2])  # one step each way, not two: 0.1996 is near enough to 0.2


def test_a_gap_past_the_longest_baseline_gives_nan_and_not_a_slope():
    t = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    slope, _ = central_difference(t, t, 0.2, 1.0)
    assert np.isnan(slope[[1, 2, 3, 4]]).sum() >= 2
    assert np.isnan(central_difference(t[2:4], t[2:4], 0.2, 1.0)[0]).all()
    np.testing.assert_allclose(slope[[0, 5]], 1.0)


def test_too_few_samples_to_span_the_baseline_give_nan():
    t = times(2)
    assert np.isnan(central_difference(t, t, 0.2, 1.0)[0]).all()


def test_a_nan_input_spoils_only_the_pairs_it_is_in():
    t = times(9)
    values = 2.0 * t
    values[4] = np.nan
    slope, _ = central_difference(values, t, 0.2, 1.0)
    assert np.isnan(slope).tolist() == [False, False, False, True, False, True, False, False, False]


def test_columns_are_differenced_separately():
    t = times(10)
    slope, _ = central_difference(np.column_stack([t, -3.0 * t]), t, 0.2, 1.0)
    np.testing.assert_allclose(slope, np.tile([1.0, -3.0], (10, 1)))


# EgoPath

def straight(x) -> np.ndarray:
    return np.column_stack([x, np.zeros_like(x)])


@pytest.mark.parametrize("sigma_m", [0.0, 0.002, 0.005, 0.02])
@pytest.mark.parametrize("seed", range(4))
def test_stopping_does_not_turn_pose_noise_into_meters_of_lateral_error(sigma_m, seed):
    """The last 5 cm step points anywhere once the ego stops, and 46 m on that's meters."""
    t = times(200)
    noise = np.random.default_rng(seed).normal(0.0, sigma_m, (200, 2))
    points = straight(np.where(t < 10.0, 10.0 * t - 0.5 * t * t, 50.0)) + noise
    path = EgoPath(points, 0.0, 0.0)
    s, lateral = path.locate(np.vstack([points[-1:], [[96.0, 0.0]]]))
    assert abs(lateral[1]) < 0.1 and s[1] - s[0] == pytest.approx(46.0, abs=1.0)


@pytest.mark.parametrize("drift", [(0.0, 0.06), (0.0, -0.3), (0.05, 0.05), (-0.06, 0.0), (0.5, 0.5)],
                         ids=["sideways", "more_sideways", "diagonal", "rolled_back", "near_a_meter"])
def test_a_parked_ego_keeps_its_heading_whatever_its_pose_drifts(drift):
    points = np.linspace([0.0, 0.0], drift, 200)
    path = EgoPath(points, 0.0, 0.0)
    assert path.length_m == 0.0
    s, lateral = path.locate([[10.0, 0.0], [0.0, 3.7], [0.0, -3.7], [-20.0, 0.0]])
    np.testing.assert_allclose(s, [10.0, 0.0, 0.0, -20.0], atol=1e-9)
    np.testing.assert_allclose(lateral, [0.0, 3.7, -3.7, 0.0], atol=1e-9)


def test_a_parked_ego_stays_parked_however_much_its_jitter_adds_up_to():
    """It never gets far, but enough 5 cm steps of noise add up to meters of arc."""
    points = np.random.default_rng(0).normal(0.0, 0.03, (200, 2))
    assert EgoPath(points, 0.0, 0.0).length_m == 0.0


def test_a_small_roll_back_at_the_end_keeps_the_extension_forward():
    points = straight(np.r_[np.linspace(0.0, 100.0, 150), np.linspace(100.0, 99.9, 20)])
    s, lateral = EgoPath(points, 0.0, 0.0).locate([[130.0, 0.0], [130.0, 2.0]])
    np.testing.assert_allclose(lateral, [0.0, 2.0], atol=1e-9)
    assert s[0] == pytest.approx(130.0, abs=0.5)


def test_both_extensions_follow_their_own_end_headings():
    """A path that turns left through 90 degrees: east at its start, north at its end."""
    arc = np.linspace(0.0, np.pi / 2.0, 100)
    points = np.column_stack([50.0 * np.sin(arc), 50.0 - 50.0 * np.cos(arc)])
    path = EgoPath(points, 0.0, np.pi / 2.0)
    s, lateral = path.locate([[50.0, 80.0], [-30.0, 0.0]])
    np.testing.assert_allclose(lateral, 0.0, atol=1e-9)
    assert s[0] == pytest.approx(path.length_m + 30.0, abs=0.01) and s[1] == pytest.approx(-30.0, abs=1e-9)


@pytest.mark.parametrize("yaw, turn", [(0.0, 1.0), (0.0, -1.0), (2.0, 1.0), (2.0, -1.0), (3.1, -1.0)])
def test_left_of_travel_is_positive_on_either_curve(yaw, turn):
    """The synthetic road only curves left, so a sign that fails on a right-hand curve passes everything else."""
    arc = np.linspace(0.0, 0.5, 200)
    local = np.column_stack([100.0 * np.sin(arc), turn * 100.0 * (1.0 - np.cos(arc))])
    rotation = rot_z(yaw)[:2, :2]
    path = EgoPath(local @ rotation.T, yaw, yaw + turn * 0.5)
    middle, normal = local[100], np.array([-np.sin(arc[100] * turn), np.cos(arc[100] * turn)])
    _, lateral = path.locate(np.vstack([middle + 3.0 * normal, middle - 3.0 * normal]) @ rotation.T)
    np.testing.assert_allclose(lateral, [3.0, -3.0], atol=0.01)


def test_outside_a_bend_the_offset_is_the_distance_to_the_corner():
    """Nearest to the corner itself, the offset isn't square to either piece, so a cross product would understate it."""
    path = EgoPath(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, -10.0]]), 0.0, -np.pi / 2.0)
    _, lateral = path.locate([[13.0, 4.0]])
    assert lateral[0] == pytest.approx(5.0)


def test_s_counts_from_the_first_pose_and_runs_along_the_path():
    path = EgoPath(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]), 0.0, np.pi / 2.0)
    s, _ = path.locate([[0.0, 0.0], [5.0, 1.0], [10.0, 5.0], [10.0, 40.0], [-7.0, 0.0]])
    np.testing.assert_allclose(s, [0.0, 5.0, 15.0, 50.0, -7.0])
    assert path.length_m == 20.0


def test_an_ego_creeping_in_tiny_steps_keeps_its_path():
    """Each step is under the 5 cm limit, and no step may be dropped against its neighbor, or the path would vanish."""
    path = EgoPath(straight(np.arange(0.0, 20.0, 0.01)), 0.0, 0.0)
    assert path.length_m == pytest.approx(20.0, abs=0.1)
    s, lateral = path.locate([[12.0, 1.0]])
    assert s[0] == pytest.approx(12.0, abs=0.06) and lateral[0] == pytest.approx(1.0)


def test_repeated_poses_make_no_empty_pieces():
    points = straight(np.repeat(np.arange(0.0, 50.0, 1.0), 3))
    s, lateral = EgoPath(points, 0.0, 0.0).locate([[20.5, 2.0]])
    assert (s[0], lateral[0]) == pytest.approx((20.5, 2.0))


def test_a_single_pose_is_a_path():
    s, lateral = EgoPath(np.array([[5.0, 5.0]]), np.pi / 2.0, np.pi / 2.0).locate([[5.0, 15.0], [2.0, 5.0]])
    np.testing.assert_allclose(s, [10.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(lateral, [0.0, 3.0], atol=1e-9)


# The ego table

@pytest.mark.parametrize("dropped", [(), (20,), (20, 41)], ids=["no_drop", "one_drop", "two_drops"])
def test_constant_braking_reads_true_on_every_row(dropped):
    """A second pass over the row times would read -4.8 on the first rows and -6.5 beside a drop."""
    t = times(60, dropped)
    ego, _ = build_ego_table(poses_at(t, 20.0 * t - 3.0 * t * t), MOTION)
    np.testing.assert_allclose(ego["a_long_mps2"], -6.0, atol=1e-6)
    np.testing.assert_allclose(ego["a_lat_mps2"], 0.0, atol=1e-6)


def test_acceleration_rotates_into_the_egos_axes():
    t = times(60)
    ego, _ = build_ego_table(poses_at(t, 0.0, 20.0 * t - 3.0 * t * t, np.pi / 2.0), MOTION)
    np.testing.assert_allclose(ego["a_long_mps2"], -6.0, atol=1e-6)
    np.testing.assert_allclose(ego["a_lat_mps2"], 0.0, atol=1e-6)


def test_acceleration_uses_its_own_longer_baseline():
    """One frame's bump in speed reaches 3 rows away over 0.5 s, and only 1 over 0.2 s."""
    t = times(60)
    speed_bump = np.zeros(60)
    speed_bump[30:] = 0.1
    ego, _ = build_ego_table(poses_at(t, 10.0 * t + speed_bump), MOTION)
    disturbed = np.flatnonzero(np.abs(ego["a_long_mps2"]) > 1e-6)
    assert disturbed.min() <= 26 and disturbed.max() >= 33


def test_velocity_and_speed_come_from_positions_in_world_axes():
    t = times(40)
    ego, _ = build_ego_table(poses_at(t, 3.0 * t, 4.0 * t, 1.0), MOTION)
    np.testing.assert_allclose(ego["v_world_x_mps"], 3.0)
    np.testing.assert_allclose(ego["v_world_y_mps"], 4.0)
    np.testing.assert_allclose(ego["speed_mps"], 5.0)


def test_positions_are_relative_to_the_first_pose():
    t = times(20)
    ego, _ = build_ego_table(poses_at(t, 5000.0 + t, -7000.0, z=80.0), MOTION)
    assert (ego["pos_local_x_m"].iloc[0], ego["pos_local_y_m"].iloc[0], ego["pos_local_z_m"].iloc[0]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("yaw_rate", [0.3, -0.3])
def test_yaw_rate_has_no_spike_where_yaw_crosses_pi(yaw_rate):
    """Westbound, atan2 jumps by 2 pi, which would read as 31 rad/s with no unwrap."""
    t = times(100)
    yaw = 3.0 * np.sign(yaw_rate) + yaw_rate * t
    assert (np.abs(yaw) > np.pi).any()
    arc_x, arc_y = np.sin(yaw) / yaw_rate, -np.cos(yaw) / yaw_rate
    ego, _ = build_ego_table(poses_at(t, 10.0 * arc_x, 10.0 * arc_y, yaw), MOTION)
    np.testing.assert_allclose(ego["yaw_rate_radps"], yaw_rate, atol=1e-6)
    assert ego["yaw_rad"].between(-np.pi, np.pi).all()


def test_path_s_follows_the_drive_across_a_stop():
    t = times(100)
    x = np.where(t < 5.0, 10.0 * t - t * t, 25.0)
    ego, path = build_ego_table(poses_at(t, x), MOTION)
    np.testing.assert_allclose(ego["path_s_m"], x, atol=0.06)
    assert path.length_m == pytest.approx(25.0, abs=0.06)


# The tracks table

def driving_north(n=60, speed=10.0) -> Poses:
    t = times(n)
    return poses_at(t, 0.0, speed * t, np.pi / 2.0)


def test_labels_rotate_into_the_egos_axes_the_right_way():
    """Facing north, a north-moving object is moving forward, and an east-moving one to the right."""
    poses = driving_north()
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), speed=(5.0, 7.0)))
    np.testing.assert_allclose(tracks["v_abs_ego_x_mps"], 7.0)
    np.testing.assert_allclose(tracks["v_abs_ego_y_mps"], -5.0)


def test_relative_velocity_subtracts_the_egos_own():
    poses = driving_north(speed=10.0)
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), speed=(0.0, 7.0)))
    np.testing.assert_allclose(tracks["v_rel_ego_x_mps"], -3.0)
    np.testing.assert_allclose(tracks["v_rel_ego_y_mps"], 0.0, atol=1e-9)


def test_relative_velocity_takes_the_ego_at_the_same_frame():
    """A frame late would be off by a tenth of a second of 6 m/s2."""
    t = times(60)
    poses = poses_at(t, 30.0 * t - 3.0 * t * t, 0.0, 0.0)
    ego, _ = build_ego_table(poses, MOTION)
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), speed=(0.0, 0.0)))
    np.testing.assert_allclose(tracks["v_rel_ego_x_mps"], -ego["v_world_x_mps"], atol=1e-9)


def test_absolute_acceleration_rotates_like_velocity():
    """An object heading north and speeding up, seen from an ego facing north: forward, not sideways."""
    poses = driving_north()
    t = times(60)
    boxes = box_rows(poses, (30.0, 0.0, 1.0)).assign(speed_x=0.0, speed_y=5.0 + 2.0 * t)
    tracks = tracks_of(poses, boxes)
    np.testing.assert_allclose(tracks["a_abs_ego_x_mps2"], 2.0, atol=1e-9)
    np.testing.assert_allclose(tracks["a_abs_ego_y_mps2"], 0.0, atol=1e-9)


def test_a_label_wins_over_the_difference_for_absolute_velocity():
    """The box sits still in the vehicle frame, so its difference is the ego's velocity. The label says otherwise."""
    poses = driving_north()
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), speed=(0.0, 3.0)))
    assert set(tracks["v_source"]) == {"label"}
    np.testing.assert_allclose(tracks["v_abs_ego_x_mps"], 3.0)
    np.testing.assert_allclose(tracks["v_diff_world_y_mps"], 10.0)


def test_differences_fill_in_and_say_so():
    poses = driving_north()
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0)))
    assert set(tracks["v_source"]) == {"central_difference"}
    np.testing.assert_allclose(tracks["v_abs_ego_x_mps"], 10.0)


def test_an_object_with_too_short_a_track_has_no_velocity():
    poses = driving_north(1)
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0)))
    assert set(tracks["v_source"]) == {"none"} and tracks["v_world_x_mps"].isna().all()
    assert not tracks["stationary"].any()  # unknown speed isn't stopped


@pytest.mark.parametrize("dropped", [(), (20,)], ids=["no_drop", "one_drop"])
def test_object_acceleration_from_differences_reads_true_on_every_row(dropped):
    t = times(60, dropped)
    poses = poses_at(t, 0.0, 10.0 * t, np.pi / 2.0)
    boxes = box_rows(poses, np.column_stack([5.0 * t + 1.0 * t * t, np.full(len(t), 30.0), np.zeros(len(t))]))
    tracks = tracks_of(poses, boxes)
    np.testing.assert_allclose(tracks["a_abs_ego_x_mps2"], 2.0, atol=1e-6)
    np.testing.assert_allclose(tracks["a_abs_ego_y_mps2"], 0.0, atol=1e-6)


def test_object_acceleration_uses_its_own_longer_baseline():
    poses = driving_north()
    t = times(60)
    boxes = box_rows(poses, (30.0, 0.0, 1.0)).assign(speed_x=0.0, speed_y=np.where(np.arange(60) < 30, 5.0, 6.0))
    disturbed = np.flatnonzero(np.abs(tracks_of(poses, boxes)["a_abs_ego_x_mps2"]) > 1e-9)
    assert disturbed.max() - disturbed.min() == 5  # six rows see the step over 0.5 s, and two over 0.2 s


def test_differences_run_per_object_over_its_own_rows():
    """Rows from two objects interleave, and differencing across them would read a 30 m jump every step."""
    poses = driving_north()
    boxes = pd.concat([box_rows(poses, (30.0, 0.0, 1.0), "A"), box_rows(poses, (-30.0, 0.0, 1.0), "B")])
    tracks = tracks_of(poses, boxes.sort_values([TIME, OBJECT], ignore_index=True))
    np.testing.assert_allclose(tracks["speed_mps"], 10.0)


@pytest.mark.parametrize("yaw, heading", [(3.0, 0.5), (-3.0, -0.5), (3.1, 3.1)])
def test_world_yaw_is_wrapped(yaw, heading):
    poses = poses_at(times(5), 0.0, 0.0, yaw)
    tracks = tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), heading=heading))
    assert tracks["yaw_world_rad"].between(-np.pi, np.pi, inclusive="left").all()
    np.testing.assert_allclose(np.cos(tracks["yaw_world_rad"]), np.cos(yaw + heading))


def test_a_stationary_object_needs_a_known_speed_under_the_threshold():
    poses = driving_north()
    tracks = pd.concat([tracks_of(poses, box_rows(poses, (30.0, 0.0, 1.0), speed=speed))
                        for speed in ((0.1, 0.1), (0.2, 0.0))])
    assert tracks["stationary"].tolist() == [True] * 60 + [False] * 60


def test_a_lead_in_the_ego_lane_reads_zero_after_the_ego_stops_with_noisy_poses():
    """The whole table, crossing pi westbound and then stopped, with noise on poses and boxes."""
    rng = np.random.default_rng(0)
    t = times(200)
    s = np.where(t < 10.0, 12.0 * t - 0.6 * t * t, 60.0)
    yaw = np.pi + 0.0
    x = -s + rng.normal(0.0, 0.002, 200)
    y = rng.normal(0.0, 0.002, 200)
    jitter = rng.normal(0.0, 1e-4, 200)  # atan2 lands either side of pi from frame to frame
    poses = poses_at(t, x, y, yaw + jitter)
    lead = np.array([-100.0, 0.0, 0.8])
    centers = world_to_vehicle(poses, lead) + rng.normal(0.0, 0.02, (200, 3))
    tracks = tracks_of(poses, box_rows(poses, centers, heading=-jitter))

    ahead = tracks[tracks["t_s"] > 12.0]
    assert np.abs(ahead["lateral_offset_m"]).max() < 0.1
    assert (ahead["path_clearance_m"] == 0.0).all()
    assert (ahead["along_path_m"] - 40.0).abs().max() < 0.2
    ego, _ = build_ego_table(poses, MOTION)
    assert np.nanmax(np.abs(ego["yaw_rate_radps"])) < 0.01


def test_along_path_is_relative_to_the_ego_now():
    poses = driving_north(60)
    tracks = tracks_of(poses, box_rows(poses, (30.0, -2.0, 1.0)))
    np.testing.assert_allclose(tracks["along_path_m"], 30.0, atol=1e-6)
    np.testing.assert_allclose(tracks["lateral_offset_m"], -2.0, atol=1e-6)


def test_clearance_is_zero_astride_the_path_and_the_nearest_corner_otherwise():
    poses = driving_north()
    on, beside = (tracks_of(poses, box_rows(poses, (20.0, y, 1.0))) for y in (0.5, 3.7))
    assert (on["path_clearance_m"] == 0.0).all()
    np.testing.assert_allclose(beside["path_clearance_m"], 3.7 - 1.9 / 2.0, atol=1e-6)


def test_front_gap_is_measured_from_the_bumper_to_the_nearest_corner():
    poses = driving_north()
    gaps = tracks_of(poses, box_rows(poses, (20.0, 0.0, 1.0)), front_m=4.0)["front_gap_m"]
    np.testing.assert_allclose(gaps, 20.0 - 4.6 / 2.0 - 4.0)
    assert tracks_of(poses, box_rows(poses, (5.0, 0.0, 1.0)), front_m=4.0)["front_gap_m"].isna().all()


def test_samples_past_the_paths_ends_say_so():
    """Their offsets are against a straight line that the road need not follow."""
    t = times(100)
    poses = poses_at(t, 2.0 * t, 0.0, 0.0)  # the path runs from 0 to about 20 m
    rows = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [10.0, 3.0, 1.0])))
    assert rows["on_driven_path"].all()
    rows = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [60.0, 3.0, 1.0])))
    assert not rows["on_driven_path"].any()
    rows = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [-5.0, 3.0, 1.0])))
    assert not rows["on_driven_path"].any()


def test_an_ego_that_never_moves_leaves_every_object_off_its_path():
    poses = poses_at(times(50), 0.0, 0.0, 0.0)
    assert not tracks_of(poses, box_rows(poses, (20.0, 0.0, 1.0)))["on_driven_path"].any()


def overpass_scene(height_m: float) -> pd.DataFrame:
    """A car at the same xy as the path halfway along it, `height_m` above the road."""
    t = times(100)
    poses = poses_at(t, 10.0 * t, 0.0, 0.0)
    return tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [50.0, 0.0, height_m])))


@pytest.mark.parametrize("height_m, cleared", [(8.5, False), (4.5, False), (1.0, True), (3.5, True), (-5.0, False)])
def test_a_box_on_another_level_has_no_clearance_to_the_path(height_m, cleared):
    rows = overpass_scene(height_m)
    assert rows["path_clearance_m"].notna().all() == cleared
    assert rows["lateral_offset_m"].abs().max() < 1e-6  # it's still directly above, and that stays visible


def test_the_height_gate_follows_the_ego_up_a_hill():
    """The ego climbs 20 m, so one height is level with it near the top, and far above it at the bottom."""
    t = times(100)
    poses = poses_at(t, 10.0 * t, 0.0, 0.0, 2.0 * t)
    near_top = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [90.0, 0.0, 19.0])))
    assert near_top["path_clearance_m"].notna().all()
    near_bottom = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [10.0, 0.0, 19.0])))
    assert near_bottom["path_clearance_m"].isna().all()


def test_past_the_paths_end_a_hill_is_not_taken_for_a_level():
    """The ego stopped, and the road ahead climbs. Its height there is unknown, so nothing can rule the car out."""
    t = times(100)
    poses = poses_at(t, 10.0 * t, 0.0, 0.0)
    rows = tracks_of(poses, box_rows(poses, world_to_vehicle(poses, [150.0, 0.0, 8.0])))
    assert not rows["on_driven_path"].any() and rows["path_clearance_m"].notna().all()


def test_aliases_count_per_prefix_by_first_appearance_then_id():
    boxes = pd.DataFrame({TIME: [2, 2, 1, 3], OBJECT: ["b", "a", "z", "p"], "type": [1, 1, 1, 2]})
    assert assign_aliases(boxes) == {"z": "V001", "a": "V002", "b": "V003", "p": "P001"}

from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agatha.columns import (
    CAMERA, CAMERA_CALIBRATION, CAMERA_IDS, CAMERA_IMAGE, CAMERA_NAMES, LIDAR_BOX, OBJECT, PROJECTED_BOX, SEGMENT,
    STAGE1_COMPONENTS, SYNCED_BOX, TIMESTAMP, VEHICLE_POSE, col,
)
from agatha.geometry import (
    BOX_EDGES, box_corners, box_rectangle, camera_from_label, project_camera_points, rot_z, se3, transform_from_list,
    transform_points, wireframe, yaw_of,
)
from agatha.loader import (
    check_segment, component_path, load_cameras, load_image_metadata, load_lidar_boxes, load_poses,
    load_projected_boxes, load_synced_boxes, read_jpegs,
)
from agatha.report import VOCABULARY
from agatha.synthetic import IMAGE_OFFSET_S, LABEL_OFFSET_S, OBJECT_IDS, write_synthetic_segment

FRONT, FRONT_LEFT, FRONT_RIGHT = CAMERA_IDS["FRONT"], CAMERA_IDS["FRONT_LEFT"], CAMERA_IDS["FRONT_RIGHT"]
CAMERAS = (FRONT, FRONT_LEFT, FRONT_RIGHT)
ROAD_CENTER_XY = (-4000.0, 11500.0)
ROAD_RADIUS_M = 200.0


@pytest.fixture(scope="module")
def segment(synthetic_root):
    root, ref = synthetic_root
    poses = load_poses(root, ref)
    images = load_image_metadata(root, ref, CAMERAS)
    return SimpleNamespace(
        root=root, ref=ref, poses=poses,
        pose_at=dict(zip(poses.timestamps.tolist(), poses.world_from_vehicle)),
        image_pose_at={(row.frame_timestamp_micros, row.camera): transform_from_list(row.world_from_image_vehicle)
                       for row in images.itertuples()},
        cameras=load_cameras(root, ref, CAMERAS),
        images=images,
        boxes=load_lidar_boxes(root, ref),
        synced=load_synced_boxes(root, ref),
        projected=load_projected_boxes(root, ref, CAMERAS),
    )


def rows_of(table, name: str):
    return table[table["laser_object_id"] == OBJECT_IDS[name]].reset_index(drop=True)


def in_camera(rows, camera: int):
    return rows[rows["camera"] == camera]


def centers(rows) -> np.ndarray:
    return rows[["center_x", "center_y", "center_z"]].to_numpy()


def world_centers(segment, rows) -> np.ndarray:
    return np.vstack([transform_points(segment.pose_at[ts], c)
                      for ts, c in zip(rows["frame_timestamp_micros"].tolist(), centers(rows))])


def road_offset(world: np.ndarray) -> np.ndarray:
    """Meters left of the ego's path, which is a circle."""
    return ROAD_RADIUS_M - np.linalg.norm(world[:, :2] - ROAD_CENTER_XY, axis=1)


def corners_in_camera(segment, row, camera: int) -> np.ndarray:
    """A label row's corners, through camera_from_label, as every real box goes."""
    ts = row.frame_timestamp_micros
    cam = segment.cameras[camera]
    camera_from_frame = camera_from_label(segment.pose_at[ts], segment.image_pose_at[ts, camera], cam)
    corners = box_corners([row.center_x, row.center_y, row.center_z], [row.size_x, row.size_y, row.size_z], row.heading)
    return transform_points(camera_from_frame, corners)


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)


def in_frame(uv: np.ndarray) -> np.ndarray:
    return (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1280)


def central_difference(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Over two steps, on interior samples: what the pipeline's differences will see."""
    return (values[2:] - values[:-2]) / (t_s[2:] - t_s[:-2])[:, None]


def read_file(segment, tag: str, columns=None) -> pa.Table:
    return pq.read_table(component_path(segment.root, segment.ref, tag), columns=columns)


def test_every_component_is_in_the_real_layout_and_passes_the_schema_check(segment):
    assert segment.root.joinpath("training", "lidar_box", "synthetic_cut_in.parquet").exists()
    checks = check_segment(segment.root, segment.ref)
    assert set(checks) == set(STAGE1_COMPONENTS)
    for check in checks.values():
        assert check.ok and check.missing_optional == []
        assert check.extra == 1  # the `index` column, which proves extras are tolerated


@pytest.mark.parametrize("tag, column, expected", [
    (VEHICLE_POSE, TIMESTAMP, pa.int64()),
    (CAMERA_IMAGE, CAMERA, pa.int8()),
    (CAMERA_IMAGE, col(CAMERA_IMAGE, "image"), pa.binary()),
    (CAMERA_IMAGE, col(CAMERA_IMAGE, "velocity.linear_velocity.x"), pa.float32()),
    (CAMERA_IMAGE, col(CAMERA_IMAGE, "pose.transform"), pa.list_(pa.float64(), 16)),
    (CAMERA_CALIBRATION, col(CAMERA_CALIBRATION, "extrinsic.transform"), pa.list_(pa.float64(), 16)),
    (CAMERA_CALIBRATION, col(CAMERA_CALIBRATION, "width"), pa.int32()),
    (CAMERA_CALIBRATION, col(CAMERA_CALIBRATION, "rolling_shutter_direction"), pa.int8()),
    (VEHICLE_POSE, col(VEHICLE_POSE, "world_from_vehicle.transform"), pa.list_(pa.float64(), 16)),
    (LIDAR_BOX, OBJECT, pa.string()),
    (LIDAR_BOX, col(LIDAR_BOX, "type"), pa.int8()),
    (LIDAR_BOX, col(LIDAR_BOX, "speed.x"), pa.float64()),
    (LIDAR_BOX, col(LIDAR_BOX, "num_top_lidar_points_in_box"), pa.int64()),
    (SYNCED_BOX, col(SYNCED_BOX, "most_visible_camera_name"), pa.int8()),
    (PROJECTED_BOX, col(PROJECTED_BOX, "type"), pa.int8()),
])
def test_column_types_match_the_real_files(segment, tag, column, expected):
    assert pq.read_schema(component_path(segment.root, segment.ref, tag)).field(column).type == expected


def test_row_order_is_shuffled_differently_per_component(segment):
    poses = read_file(segment, VEHICLE_POSE, [TIMESTAMP])[TIMESTAMP].to_numpy()
    images = read_file(segment, CAMERA_IMAGE, [TIMESTAMP, CAMERA]).to_pandas()
    fronts = images.loc[images[CAMERA] == FRONT, TIMESTAMP].to_numpy()
    assert (np.diff(poses) < 0).any()
    assert not np.array_equal(fronts, poses)  # the same keys in a different order
    assert np.array_equal(np.sort(fronts), np.sort(poses))


def test_row_order_differs_between_components_of_the_same_length(tmp_path):
    # With one camera, these two line up row for row unless each has its own order.
    ref = write_synthetic_segment(tmp_path, n_frames=12, cameras=("FRONT",))
    poses, images = (pq.read_table(component_path(tmp_path, ref, tag), columns=[TIMESTAMP])[TIMESTAMP].to_pylist()
                     for tag in (VEHICLE_POSE, CAMERA_IMAGE))
    assert poses != images and sorted(poses) == sorted(images)


def test_index_column_counts_file_rows(segment):
    index = read_file(segment, LIDAR_BOX, ["index"])["index"].to_numpy()
    assert np.array_equal(index, np.arange(len(segment.boxes)))


def test_timing_is_10_hz_with_jitter_and_two_dropped_frames(segment):
    assert len(segment.poses.timestamps) == 199
    dt_ms = np.diff(segment.poses.timestamps) / 1e3
    gaps = dt_ms > 150.0
    assert gaps.sum() == 2
    assert np.abs(dt_ms[gaps] - 200.0).max() < 1.0
    regular = dt_ms[~gaps]
    assert np.abs(regular - 100.0).max() < 1.0 and len(np.unique(regular)) > 10


def test_every_key_joins_exactly(segment):
    frames = set(segment.poses.timestamps.tolist())
    assert set(segment.boxes["frame_timestamp_micros"]) == frames
    assert set(segment.images["frame_timestamp_micros"]) == frames
    labeled = set(zip(segment.boxes["frame_timestamp_micros"], segment.boxes["laser_object_id"]))
    assert len(labeled) == len(segment.boxes)
    assert set(zip(segment.synced["frame_timestamp_micros"], segment.synced["laser_object_id"])) <= labeled
    assert set(zip(segment.projected["frame_timestamp_micros"], segment.projected["laser_object_id"])) <= labeled
    assert not segment.projected.duplicated(["frame_timestamp_micros", "camera", "laser_object_id"]).any()
    assert not segment.images.duplicated(["frame_timestamp_micros", "camera"]).any()


def test_calibration_is_waymos_front_with_the_others_yawed(segment):
    front, left, right = (segment.cameras[camera] for camera in CAMERAS)
    assert front.intrinsic[0] == 2055.55614936 and (front.width, front.height) == (1920, 1280)
    assert front.extrinsic[0, 3] == 1.53891424471
    assert front.rolling_shutter_direction == 2  # LEFT_TO_RIGHT, as in Waymo's fixture
    np.testing.assert_allclose(left.extrinsic, se3(rot_z(np.pi / 4), np.zeros(3)) @ front.extrinsic)
    np.testing.assert_allclose(right.extrinsic, se3(rot_z(-np.pi / 4), np.zeros(3)) @ front.extrinsic)
    assert left.extrinsic[1, 0] > 0.7 and right.extrinsic[1, 0] < -0.7  # left looks left


def test_world_origin_is_remote(segment):
    assert np.abs(segment.poses.world_from_vehicle[:, :2, 3]).min() > 1000.0


def test_ego_decelerates_along_a_gentle_curve(segment):
    t_s = (segment.poses.timestamps - segment.poses.timestamps[0]) / 1e6
    position = segment.poses.world_from_vehicle[:, :3, 3]
    speed = np.linalg.norm(central_difference(position, t_s), axis=1)
    assert (speed[0], speed[-1]) == pytest.approx((14.0, 6.0), abs=0.2)
    assert (np.diff(speed) < 0).mean() > 0.95

    yaw = np.unwrap([yaw_of(pose) for pose in segment.poses.world_from_vehicle])
    assert (yaw[-1] - yaw[0]) / t_s[-1] == pytest.approx(0.05, abs=0.005)


@pytest.mark.parametrize("camera, offset_ms", [(FRONT_LEFT, 37.5), (FRONT, 50.0), (FRONT_RIGHT, 62.5)])
def test_image_poses_sit_at_their_cameras_nominal_offsets(segment, camera, offset_ms):
    images = in_camera(segment.images, camera)
    np.testing.assert_allclose(images["pose_timestamp_s"] * 1e3 - images["frame_timestamp_micros"] / 1e3,
                               offset_ms, atol=1e-3)


def test_vehicle_pose_is_near_fronts_pose_but_not_on_it(segment):
    # Equal poses would let code that skips camera_from_label pass on FRONT.
    ts = int(segment.poses.timestamps[50])
    gap_m = np.linalg.norm(segment.pose_at[ts][:3, 3] - segment.image_pose_at[ts, FRONT][:3, 3])
    assert 0.005 < gap_m < 0.1


@pytest.mark.parametrize("camera", CAMERAS)
def test_image_pose_is_the_ego_at_that_time(segment, camera):
    # The offset from vehicle_pose along the ego's own motion is what camera_from_label corrects.
    dt_s = IMAGE_OFFSET_S[CAMERA_NAMES[camera]] - LABEL_OFFSET_S
    for row in in_camera(segment.images, camera).iloc[::37].itertuples():
        frame_pose = segment.pose_at[row.frame_timestamp_micros]
        shift = transform_from_list(row.world_from_image_vehicle)[:3, 3] - frame_pose[:3, 3]
        velocity = [row.linear_velocity_x_mps, row.linear_velocity_y_mps, row.linear_velocity_z_mps]
        np.testing.assert_allclose(shift, np.array(velocity) * dt_s, atol=2e-3)


def test_ego_velocity_columns_match_the_poses(segment):
    front = in_camera(segment.images, FRONT)
    t_s = front["pose_timestamp_s"].to_numpy() - front["pose_timestamp_s"].iloc[0]
    poses = np.stack([transform_from_list(p) for p in front["world_from_image_vehicle"]])
    measured = central_difference(poses[:, :3, 3], t_s)
    reported = front[["linear_velocity_x_mps", "linear_velocity_y_mps", "linear_velocity_z_mps"]].to_numpy()[1:-1]
    assert np.abs(measured - reported).max() < 0.05
    assert (front["linear_velocity_z_mps"] == 0.0).all()


def test_rolling_shutter_times_are_coherent(segment):
    images = segment.images
    readout = images["readout_done_time_s"] - images["trigger_time_s"] - images["shutter_s"]
    np.testing.assert_allclose(readout, 0.044, atol=1e-5)
    lag = images["pose_timestamp_s"] - (images["trigger_time_s"] + images["readout_done_time_s"]) / 2.0
    np.testing.assert_allclose(lag, 0.0043, atol=1e-5)
    assert images["shutter_s"].between(0.001, 0.010).all() and images["shutter_s"].std() > 0.0


def test_speed_is_in_world_axes(segment):
    # Check 5 on real data: world axes must beat the vehicle-axes reading by a wide margin.
    rows = rows_of(segment.boxes, "lead")
    t_s = rows["frame_timestamp_micros"].to_numpy() / 1e6
    measured = central_difference(world_centers(segment, rows), t_s)[:, :2]
    labeled = rows[["speed_x", "speed_y"]].to_numpy()[1:-1]
    known = np.isfinite(labeled[:, 0])

    in_world = np.linalg.norm(measured - labeled, axis=1)[known]
    assert np.median(in_world) < 0.02

    yaw = np.array([yaw_of(segment.pose_at[ts]) for ts in rows["frame_timestamp_micros"]])[1:-1]
    rotated = np.stack([rot_z(a)[:2, :2] @ v for a, v in zip(yaw, labeled)])
    assert np.median(np.linalg.norm(measured - rotated, axis=1)[known]) > 5.0


def test_differencing_vehicle_frame_positions_invents_lateral_motion(segment):
    # The reason for the curve: in a yawing frame, d(y)/dt = true lateral velocity - yaw_rate * x.
    rows = rows_of(segment.boxes, "lead")
    known = rows["speed_x"].notna().to_numpy()
    t_s = rows["frame_timestamp_micros"].to_numpy() / 1e6
    naive = central_difference(centers(rows), t_s)[:, 1]

    poses = np.stack([segment.pose_at[ts] for ts in rows["frame_timestamp_micros"]])
    ego_velocity = central_difference(poses[:, :3, 3], t_s)
    yaw = np.unwrap([yaw_of(pose) for pose in poses])
    yaw_rate = (yaw[2:] - yaw[:-2]) / (t_s[2:] - t_s[:-2])
    true_lateral = np.array([(pose[:3, :3].T @ (np.append(v, 0.0) - e))[1] for pose, v, e in
                             zip(poses[1:-1], rows[["speed_x", "speed_y"]].to_numpy()[1:-1], ego_velocity)])

    false_term = (naive - true_lateral)[known[1:-1]]
    assert np.abs(false_term).min() > 1.0
    np.testing.assert_allclose(false_term, -(yaw_rate * centers(rows)[1:-1, 0])[known[1:-1]], atol=0.15)


def test_a_few_labels_have_no_speed_and_the_rest_have_zero_z(segment):
    boxes = segment.boxes
    missing = boxes["speed_x"].isna()
    assert 0.005 < missing.mean() < 0.1
    for column in ("speed_y", "speed_z", "accel_x", "accel_y", "accel_z"):  # absent as a whole
        assert boxes[column].isna().equals(missing)
    assert (boxes.loc[~missing, "speed_z"] == 0.0).all()
    moving = boxes.loc[missing, "laser_object_id"].isin([OBJECT_IDS["cut_in"], OBJECT_IDS["lead"]])
    assert moving.any()  # or v_source's fallback would never run on a moving object


def test_acceleration_is_the_derivative_of_label_speed(segment):
    rows = rows_of(segment.boxes, "cut_in")
    rows = rows[rows["speed_x"].notna()]
    t_s = rows["frame_timestamp_micros"].to_numpy() / 1e6
    measured = central_difference(rows[["speed_x", "speed_y"]].to_numpy(), t_s)
    errors = np.abs(measured - rows[["accel_x", "accel_y"]].to_numpy()[1:-1])
    assert np.median(errors) < 1e-3
    assert errors.max() < 1.4  # a difference rounds off the 1.39 m/s2 jumps at the lane change's corners


def test_heading_is_wrapped_and_follows_the_road(segment):
    boxes = segment.boxes
    assert boxes["heading"].between(-np.pi, np.pi, inclusive="left").all()
    assert np.abs(rows_of(boxes, "beside")["heading"]).max() < 0.1  # alongside, so nearly the ego's heading
    sign = rows_of(boxes, "sign")["heading"]  # faces the traffic: about pi, and it crosses the wrap
    assert np.cos(sign).max() < -0.9 and (sign > 0).any() and (sign < 0).any()


def test_labels_stop_at_the_labeled_range(segment):
    boxes = segment.boxes
    assert np.hypot(boxes["center_x"], boxes["center_y"]).max() <= 75.0
    parked = rows_of(boxes, "parked")
    assert parked["frame_timestamp_micros"].iloc[0] > segment.poses.timestamps[0]
    assert np.hypot(parked["center_x"], parked["center_y"]).iloc[0] > 70.0  # it starts abruptly


def test_cut_in_crosses_from_the_adjacent_lane_into_the_ego_path(segment):
    rows = rows_of(segment.boxes, "cut_in")
    t_s = (rows["frame_timestamp_micros"].to_numpy() - segment.poses.timestamps[0]) / 1e6
    offset = road_offset(world_centers(segment, rows))
    assert offset[t_s < 7.0] == pytest.approx(-3.7, abs=1e-6)
    assert offset[t_s > 13.0] == pytest.approx(0.0, abs=1e-6)
    assert (np.diff(offset[(t_s > 8.0) & (t_s < 12.0)]) > 0).all()
    assert (rows["center_x"].iloc[0], rows["center_y"].iloc[0]) == pytest.approx((3.1, -3.7), abs=0.1)


def test_cut_in_starts_outside_front_and_then_enters_it(segment):
    rows = rows_of(segment.projected, "cut_in")
    first_front = in_camera(rows, FRONT)["frame_timestamp_micros"].min()
    assert first_front > segment.poses.timestamps[0]
    assert in_camera(rows, FRONT_RIGHT)["frame_timestamp_micros"].min() < first_front


def test_lead_stays_ahead_in_the_ego_lane(segment):
    rows = rows_of(segment.boxes, "lead")
    assert len(rows) == len(segment.poses.timestamps)
    assert rows["center_x"].between(44.0, 51.0).all()
    assert len(in_camera(rows_of(segment.projected, "lead"), FRONT)) == len(rows)


@pytest.mark.parametrize("name", ["parked", "sign"])
def test_stationary_objects_stay_put_in_the_world(segment, name):
    rows = rows_of(segment.boxes, name)
    world = world_centers(segment, rows)
    assert np.ptp(world, axis=0).max() < 1e-6
    assert np.ptp(rows["center_x"]) > 50.0  # while the ego sweeps past them in its own frame
    assert (rows["speed_x"].dropna() == 0.0).all() and (rows["speed_y"].dropna() == 0.0).all()


def test_pedestrian_crosses_the_ego_path(segment):
    rows = rows_of(segment.boxes, "pedestrian")
    assert set(rows["type"]) == {2}
    offset = road_offset(world_centers(segment, rows))
    assert offset.max() == pytest.approx(7.0, abs=1e-6) and offset.min() < -1.0
    speed = np.hypot(rows["speed_x"], rows["speed_y"]).dropna()
    assert speed.max() == pytest.approx(1.4, abs=0.01) and speed.iloc[0] < 0.01


def test_sign_is_a_sign(segment):
    assert set(rows_of(segment.boxes, "sign")["type"]) == {3}
    assert set(rows_of(segment.projected, "sign")["type"]) == {3}
    assert (rows_of(segment.boxes, "sign")["center_z"] > 1.0).all()


def test_one_vehicle_has_no_top_lidar_points_for_part_of_its_track(segment):
    boxes = segment.boxes
    unseen = boxes[boxes["num_top_lidar_points"] == 0]
    assert set(unseen["laser_object_id"]) == {OBJECT_IDS["occluded"]}
    t_s = (unseen["frame_timestamp_micros"] - segment.poses.timestamps[0]) / 1e6
    assert t_s.min() == pytest.approx(5.0, abs=0.2) and t_s.max() == pytest.approx(9.0, abs=0.2)
    assert (rows_of(boxes, "occluded")["num_top_lidar_points"] > 0).any()
    assert (boxes.loc[boxes["num_top_lidar_points"] > 0, "num_lidar_points"] > 0).all()


def test_beside_vehicle_is_the_ghost_box_case(segment):
    # A naive projection folds this box into FRONT. Guarded, FRONT has no row for it at all.
    assert in_camera(rows_of(segment.projected, "beside"), FRONT).empty
    front = segment.cameras[FRONT]
    t = np.linspace(0.0, 1.0, 40)[:, None]
    for row in rows_of(segment.boxes, "beside").iloc[::40].itertuples():
        corners = corners_in_camera(segment, row, FRONT)
        samples = np.vstack([corners[a] * (1.0 - t) + corners[b] * t for a, b in BOX_EDGES])
        proj = project_camera_points(samples[samples[:, 0] > 0.1], front)  # ahead of the lens, so only the fold is left
        assert in_frame(proj.uv).any() and not in_frame(proj.uv[proj.valid]).any()
        assert not any(in_frame(piece).any() for piece in wireframe(corners, front))


def test_beside_vehicle_shows_in_front_left(segment):
    assert len(in_camera(rows_of(segment.projected, "beside"), FRONT_LEFT)) == len(rows_of(segment.boxes, "beside"))


@pytest.mark.parametrize("name, camera", [("parked_left", FRONT_LEFT), ("parked_right", FRONT_RIGHT)])
def test_each_side_camera_sees_a_whole_car(segment, name, camera):
    """Check 7 only compares boxes clear of the border, and every other car is clamped in a side view."""
    rows = in_camera(rows_of(segment.projected, name), camera)
    clear_u = ((rows["center_x"] - rows["size_x"] / 2.0) > 2.0) & ((rows["center_x"] + rows["size_x"] / 2.0) < 1918.0)
    clear_v = ((rows["center_y"] - rows["size_y"] / 2.0) > 2.0) & ((rows["center_y"] + rows["size_y"] / 2.0) < 1278.0)
    assert (clear_u & clear_v).sum() >= 10


def test_oracle_matches_a_fresh_projection_on_unclamped_boxes(segment):
    # Near zero by construction, so this only proves the wiring: rows, keys, axes and poses.
    merged = segment.projected.merge(segment.boxes, on=["frame_timestamp_micros", "laser_object_id"],
                                     suffixes=("_image", ""))
    compared = 0
    for row in merged.itertuples():
        cam = segment.cameras[row.camera]
        rectangle = box_rectangle(corners_in_camera(segment, row, row.camera), cam)
        if rectangle is None:
            continue
        u_min, u_max, v_min, v_max = rectangle
        if u_min < 2.0 or v_min < 2.0 or u_max > cam.width - 2.0 or v_max > cam.height - 2.0:
            continue  # Waymo clamps these, so the rectangle says nothing about the far edge
        stored = (row.center_x_image, row.center_y_image, row.size_x_image, row.size_y_image)
        expected = ((u_min + u_max) / 2, (v_min + v_max) / 2, u_max - u_min, v_max - v_min)
        assert stored == pytest.approx(expected, abs=1e-9)
        compared += 1
    assert compared > 300


def test_oracle_is_clamped_to_the_image(segment):
    projected = segment.projected
    low_u, high_u = projected["center_x"] - projected["size_x"] / 2.0, projected["center_x"] + projected["size_x"] / 2.0
    low_v, high_v = projected["center_y"] - projected["size_y"] / 2.0, projected["center_y"] + projected["size_y"] / 2.0
    assert low_u.min() >= -1e-9 and high_u.max() <= 1920.0 + 1e-9
    assert low_v.min() >= -1e-9 and high_v.max() <= 1280.0 + 1e-9
    assert (high_u > 1920.0 - 1e-6).any() and (low_u < 1e-6).any()  # some really do hit an edge
    assert (projected["size_x"] > 0.0).all() and (projected["size_y"] > 0.0).all()


def test_oracle_omits_objects_behind_the_ego(segment):
    # parked, once the ego has passed it: labeled, but in no camera, so no row and no synced box.
    labels = rows_of(segment.boxes, "parked")
    behind = labels.loc[labels["center_x"] < -20.0, "frame_timestamp_micros"]
    assert len(behind) > 10
    assert not rows_of(segment.projected, "parked")["frame_timestamp_micros"].isin(behind).any()
    assert not rows_of(segment.synced, "parked")["frame_timestamp_micros"].isin(behind).any()


def test_synced_box_exists_exactly_when_a_projected_box_does(segment):
    synced = set(zip(segment.synced["frame_timestamp_micros"], segment.synced["laser_object_id"]))
    projected = set(zip(segment.projected["frame_timestamp_micros"], segment.projected["laser_object_id"]))
    assert synced == projected


def test_synced_camera_is_the_one_with_the_largest_rectangle(segment):
    areas = segment.projected.assign(area=segment.projected["size_x"] * segment.projected["size_y"])
    best = areas.sort_values("area").groupby(["frame_timestamp_micros", "laser_object_id"])["camera"].last()
    chosen = segment.synced.set_index(["frame_timestamp_micros", "laser_object_id"])["most_visible_camera"]
    assert (chosen.sort_index() == best.sort_index()).all()
    assert chosen.nunique() > 1  # or the choice was never really made


@pytest.mark.parametrize("name", ["cut_in", "lead", "occluded"])
def test_synced_center_is_the_label_moved_to_capture_time(segment, name):
    # In that image's vehicle frame, so the world position reads through that image's pose.
    synced = rows_of(segment.synced, name)
    labels = rows_of(segment.boxes, name).set_index("frame_timestamp_micros")
    shifts = []
    for row in synced.itertuples():
        label = labels.loc[row.frame_timestamp_micros]
        if np.isnan(label["speed_x"]):
            continue
        pose = segment.pose_at[row.frame_timestamp_micros]
        world_label = transform_points(pose, label[["center_x", "center_y", "center_z"]].to_numpy(dtype=float))[0]
        image_pose = segment.image_pose_at[row.frame_timestamp_micros, row.most_visible_camera]
        world_synced = transform_points(image_pose, [row.center_x, row.center_y, row.center_z])[0]
        dt_s = IMAGE_OFFSET_S[CAMERA_NAMES[row.most_visible_camera]] - LABEL_OFFSET_S
        expected = world_label + np.array([label["speed_x"], label["speed_y"], 0.0]) * dt_s
        np.testing.assert_allclose(world_synced, expected, atol=2e-3)
        shifts.append(np.linalg.norm(world_synced - world_label))
    assert min(shifts) > 0.01  # moving objects really are shifted


def test_synced_box_keeps_size_and_heading(segment):
    merged = segment.synced.merge(segment.boxes, on=["frame_timestamp_micros", "laser_object_id"],
                                  suffixes=("", "_label"))
    for column in ("size_x", "size_y", "size_z", "heading"):
        assert (merged[column] == merged[f"{column}_label"]).all()


def test_images_are_jpegs_of_the_declared_size(segment):
    jpegs = read_jpegs(segment.root, segment.ref, FRONT_LEFT)
    assert set(jpegs) == set(segment.poses.timestamps.tolist())
    assert decode(next(iter(jpegs.values()))).shape == (1280, 1920, 3)


def test_images_show_the_boxes_where_the_oracle_puts_them(segment):
    # Only the pedestrian is red, so the bounding box of the red pixels is its drawn extent.
    # That is a real alignment check, late in the segment when everything else is past it.
    rows = in_camera(rows_of(segment.projected, "pedestrian"), FRONT).iloc[-30::6]
    jpegs = read_jpegs(segment.root, segment.ref, FRONT, rows["frame_timestamp_micros"].tolist())
    for row in rows.itertuples():
        image = decode(jpegs[row.frame_timestamp_micros])  # BGR
        v, u = np.nonzero((image[..., 2] > image[..., 0] + 60) & (image[..., 1] < 120))
        half_u, half_v = row.size_x / 2, row.size_y / 2
        np.testing.assert_allclose([u.min(), u.max()], [row.center_x - half_u, row.center_x + half_u], atol=4.0)
        np.testing.assert_allclose([v.min(), v.max()], [row.center_y - half_v, row.center_y + half_v], atol=4.0)


def test_images_differ_by_camera_and_frame(segment):
    front = read_jpegs(segment.root, segment.ref, FRONT)
    left = read_jpegs(segment.root, segment.ref, FRONT_LEFT)
    first, later = sorted(front)[0], sorted(front)[100]
    assert front[first] != left[first] and front[first] != front[later]


def test_same_seed_writes_identical_files_and_another_seed_does_not(tmp_path):
    ref = write_synthetic_segment(tmp_path / "a", n_frames=9, seed=3)
    write_synthetic_segment(tmp_path / "b", n_frames=9, seed=3)
    write_synthetic_segment(tmp_path / "c", n_frames=9, seed=4)
    for tag in STAGE1_COMPONENTS:
        bytes_a = component_path(tmp_path / "a", ref, tag).read_bytes()
        assert bytes_a == component_path(tmp_path / "b", ref, tag).read_bytes()
        if tag != CAMERA_CALIBRATION:  # calibration has no randomness but its row order
            assert bytes_a != component_path(tmp_path / "c", ref, tag).read_bytes()


def test_arguments_shape_the_segment(tmp_path):
    ref = write_synthetic_segment(tmp_path, split="validation", context_name="synthetic_other", n_frames=12,
                                  cameras=("FRONT",))
    assert (ref.split, ref.context_name) == ("validation", "synthetic_other")
    assert all(check.ok for check in check_segment(tmp_path, ref).values())
    poses = load_poses(tmp_path, ref)
    assert len(poses.timestamps) == 12  # n_frames counts what's written
    assert (np.diff(poses.timestamps) > 150_000).sum() == 2
    assert set(load_image_metadata(tmp_path, ref, CAMERAS)["camera"]) == {FRONT}
    assert set(load_projected_boxes(tmp_path, ref, CAMERAS)["camera"]) == {FRONT}
    assert set(load_synced_boxes(tmp_path, ref)["most_visible_camera"]) == {FRONT}
    assert list(load_cameras(tmp_path, ref, CAMERAS)) == [FRONT]
    labels = pq.read_table(component_path(tmp_path, ref, LIDAR_BOX), columns=[SEGMENT])
    assert set(labels[SEGMENT].to_pylist()) == {"synthetic_other"}


def test_object_ids_have_the_real_shape_and_stay_out_of_the_report_vocabulary():
    assert len(set(OBJECT_IDS.values())) == 9
    for object_id in OBJECT_IDS.values():
        assert len(object_id) == 22 and object_id not in VOCABULARY and not object_id.islower()


def test_stage1_config_points_at_the_synthetic_segment(stage1_config, synthetic_root, tmp_path):
    root, ref = synthetic_root
    assert stage1_config.root == root and stage1_config.outputs.parent == tmp_path
    assert check_segment(stage1_config.root, ref)[LIDAR_BOX].ok
    assert stage1_config.cameras == (FRONT,)

import cv2
import numpy as np
import pytest

from agatha.geometry import (
    BOX_EDGES, CameraModel, box_corners, box_rectangle, camera_from_label, inv_se3, project_camera_points,
    rot_z, se3, transform_from_list, transform_points, true_runs, wireframe, wrap_angle, yaw_of,
)

# These constants come from Waymo's camera_model_test.cc, see NOTICE. They are no-data
# vectors: a real FRONT calibration with known answers, and none of them is a Dataset row.
INTRINSIC = (2055.55614936, 2055.55614936, 939.657469886, 641.072182194, 0.032316008498,
             -0.321412482553, 0.000793258395371, -0.000625749354133, 0.0)
EXTRINSIC = transform_from_list([
    0.999892684989, -0.00599320840002, 0.0133678704017, 1.53891424471,
    0.00604223652133, 0.999975156055, -0.0036302411765, -0.0236339408393,
    -0.0133457814992, 0.00371062343188, 0.999904056092, 2.11527057298,
    0, 0, 0, 1,
])
IMAGE_POSE = transform_from_list([
    -0.913574384152, -0.406212760482, -0.0193141875914, -4069.03497872,
    0.406637479491, -0.913082565675, -0.0304333457449, 11526.3118079,
    -0.00527303457417, -0.0356569976572, 0.999350175676, 86.504,
    0, 0, 0, 1,
])
WAYMO = CameraModel(INTRINSIC, EXTRINSIC, 1920, 1280)
VEHICLE_POINT = (21.44064, 8.32240, -1.62938)
WORLD_POINT = (-4091.97180, 11527.48092, 84.46586)
EXPECTED_UV = (100.0006, 999.9993)
EXPECTED_DEPTH = 19.99999

PINHOLE = CameraModel((1000.0, 1000.0, 960.0, 640.0, 0, 0, 0, 0, 0), np.eye(4), 1920, 1280)


def opencv_uv(points_camera, cam: CameraModel) -> np.ndarray:
    """cv2.projectPoints: the same lens formula, with no guard on the radial factor."""
    f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = cam.intrinsic
    matrix = np.array([[f_u, 0.0, c_u], [0.0, f_v, c_v], [0.0, 0.0, 1.0]])
    x, y, z = np.asarray(points_camera, dtype=np.float64).T
    uv, _ = cv2.projectPoints(np.stack([-y, -z, x], axis=1), np.zeros(3), np.zeros(3), matrix,
                              np.array([k1, k2, p1, p2, k3]))
    return uv.reshape(-1, 2)


def inside(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    return (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)


def rot_y(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_se3_inverse_roundtrip():
    T = se3(rot_z(0.7) @ rot_y(-0.1), [3.0, -2.0, 1.5])
    pts = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(transform_points(inv_se3(T), transform_points(T, pts)), pts, atol=1e-12)
    np.testing.assert_allclose(inv_se3(T) @ T, np.eye(4), atol=1e-12)


def test_transform_from_list_is_row_major():
    np.testing.assert_allclose(EXTRINSIC[:3, 3], [1.53891424471, -0.0236339408393, 2.11527057298])
    np.testing.assert_array_equal(EXTRINSIC[3], [0, 0, 0, 1])
    assert EXTRINSIC.dtype == np.float64


def test_rotation_signs():
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)  # yaw left


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5, 3.1])
def test_yaw_of(yaw):
    assert yaw_of(se3(rot_z(yaw), [100.0, -50.0, 3.0])) == pytest.approx(yaw)


@pytest.mark.parametrize("angle, wrapped", [
    (0.5, 0.5), (3.5, 3.5 - 2.0 * np.pi), (-3.5, 2.0 * np.pi - 3.5), (np.pi, -np.pi),
])
def test_wrap_angle_lands_in_the_half_open_range(angle, wrapped):
    assert wrap_angle(angle) == pytest.approx(wrapped)


def test_waymo_vector_from_the_vehicle_frame():
    camera = transform_points(WAYMO.camera_from_vehicle, VEHICLE_POINT)
    proj = project_camera_points(camera, WAYMO)
    np.testing.assert_allclose(proj.uv[0], EXPECTED_UV, atol=0.01)
    assert camera[0, 0] == pytest.approx(EXPECTED_DEPTH, abs=1e-4)
    assert proj.valid[0] and inside(proj.uv, WAYMO)[0]


def test_waymo_vector_from_the_world_through_the_image_pose():
    # World is the label frame here, so this runs the path every real box takes.
    camera = transform_points(camera_from_label(np.eye(4), IMAGE_POSE, WAYMO), WORLD_POINT)
    proj = project_camera_points(camera, WAYMO)
    np.testing.assert_allclose(proj.uv[0], EXPECTED_UV, atol=0.1)
    assert camera[0, 0] == pytest.approx(EXPECTED_DEPTH, abs=1e-3)


def test_waymo_point_behind_the_camera_is_invalid():
    # camera_model_test.cc shifts world x by 200, which lands behind this pose's camera.
    world = np.array(WORLD_POINT) + [200.0, 0.0, 0.0]
    camera = transform_points(camera_from_label(np.eye(4), IMAGE_POSE, WAYMO), world)
    assert not project_camera_points(camera, WAYMO).valid[0]
    assert camera[0, 0] < 0


def test_matches_opencv_where_the_lens_model_holds():
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(2.0, 80.0, 500), rng.uniform(-30.0, 30.0, 500), rng.uniform(-5.0, 5.0, 500)])
    proj = project_camera_points(points, WAYMO)
    assert proj.valid.sum() > 200
    np.testing.assert_allclose(proj.uv[proj.valid], opencv_uv(points, WAYMO)[proj.valid], atol=1e-6)


def test_matches_opencv_with_every_lens_term_active():
    # Waymo's fixture has k3 = 0, so it can't show that k3 is applied, or in which slot.
    cam = CameraModel((1000.0, 1100.0, 950.0, 630.0, -0.1, 0.05, 0.01, -0.02, 0.03), np.eye(4), 1920, 1280)
    rng = np.random.default_rng(1)
    points = np.column_stack([np.full(100, 1.0), rng.uniform(-0.7, 0.7, 100), rng.uniform(-0.5, 0.5, 100)])
    proj = project_camera_points(points, cam)
    assert proj.valid.mean() > 0.9
    np.testing.assert_allclose(proj.uv[proj.valid], opencv_uv(points, cam)[proj.valid], atol=1e-6)


def test_waymo_camera_axes():
    # Camera frame is x forward, y left, z up (not OpenCV's x right, y down, z forward).
    proj = project_camera_points([[10, 0, 0], [10, 1, 0], [10, 0, 1]], PINHOLE)
    np.testing.assert_allclose(proj.uv[0], [960, 640])
    np.testing.assert_allclose(proj.uv[1], [860, 640])  # left in the world -> left in the image
    np.testing.assert_allclose(proj.uv[2], [960, 540])  # up in the world -> up in the image


def test_extrinsic_is_camera_to_vehicle():
    cam = CameraModel(PINHOLE.intrinsic, se3(np.eye(3), [1.5, 0.0, 2.0]), 1920, 1280)
    camera = transform_points(cam.camera_from_vehicle, [[11.5, 0.0, 2.0]])
    np.testing.assert_allclose(camera, [[10.0, 0.0, 0.0]], atol=1e-12)
    # A point at the camera's height straight ahead lands on the principal point.
    np.testing.assert_allclose(project_camera_points(camera, cam).uv[0], [960, 640])
    np.testing.assert_allclose(WAYMO.camera_from_vehicle @ WAYMO.extrinsic, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("x", [-10.0, -0.1, 0.0, 0.05])
def test_point_behind_or_at_the_camera_is_invalid(x):
    # Off axis as well as on it: the lens guard alone would reject some of these.
    proj = project_camera_points([[x, 0.0, 0.0], [x, 0.5, 0.2]], WAYMO)
    assert not proj.valid.any()
    assert np.isfinite(proj.uv).all()  # no division by zero leaks out


def test_valid_does_not_mean_in_image():
    # Past the image edge but inside the lens model: wireframes need these points to clip.
    proj = project_camera_points([[10.0, 7.0, 0.0]], WAYMO)
    assert proj.valid[0] and not inside(proj.uv, WAYMO)[0]


def test_camera_from_label_reduces_to_camera_from_vehicle_when_poses_match():
    np.testing.assert_allclose(camera_from_label(IMAGE_POSE, IMAGE_POSE, WAYMO), WAYMO.camera_from_vehicle, atol=1e-9)


def test_camera_from_label_is_the_world_chain():
    frame_pose = IMAGE_POSE @ se3(rot_z(0.01), [-1.2, 0.1, 0.0])
    points = np.array([VEHICLE_POINT, [40.0, -3.0, 0.5]])
    through_world = transform_points(inv_se3(IMAGE_POSE @ WAYMO.extrinsic), transform_points(frame_pose, points))
    np.testing.assert_allclose(transform_points(camera_from_label(frame_pose, IMAGE_POSE, WAYMO), points),
                               through_world, atol=1e-9)


def test_camera_from_label_uses_the_image_pose_for_the_camera():
    # The ego drove 1 m between the label pose and the image pose, so a labeled point ahead
    # is 1 m closer to the camera. Swapping the two poses would make it 1 m farther.
    image_pose = IMAGE_POSE @ se3(np.eye(3), [1.0, 0.0, 0.0])
    camera = transform_points(camera_from_label(IMAGE_POSE, image_pose, PINHOLE), [10.0, 0.0, 0.0])
    np.testing.assert_allclose(camera, [[9.0, 0.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_projection_invariant_to_rigid_motion(yaw):
    """Moving both the camera mount and the points by the same transform changes nothing."""
    T = se3(rot_z(yaw), [4.0, -1.0, 0.3])
    pts = np.array([[12.0, 1.0, 0.5], [30.0, -4.0, 1.0]])
    moved = CameraModel(WAYMO.intrinsic, T @ WAYMO.extrinsic, 1920, 1280)
    camera = transform_points(moved.camera_from_vehicle, transform_points(T, pts))
    np.testing.assert_allclose(project_camera_points(camera, moved).uv,
                               project_camera_points(transform_points(WAYMO.camera_from_vehicle, pts), WAYMO).uv,
                               atol=1e-9)


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_box_beside_the_ego_draws_no_ghost(side):
    # 3 m ahead of the camera and 3.5 to 3.9 m to the side, far outside the lens model.
    corners = box_corners([3.0, side * 3.7, -1.0], [0.4, 0.4, 2.0], 0.0)
    assert corners[:, 1].min() == pytest.approx(3.5 if side > 0 else -3.9)
    assert corners[:, 1].max() == pytest.approx(3.9 if side > 0 else -3.5)

    assert inside(opencv_uv(corners, WAYMO), WAYMO).any()  # a naive projection puts it in the image
    assert not project_camera_points(corners, WAYMO).valid.any()
    assert box_rectangle(corners, WAYMO) is None
    assert wireframe(corners, WAYMO) == []


@pytest.mark.parametrize("angle_deg", range(0, 360, 45))
def test_distorted_radius_is_monotonic_where_valid(angle_deg):
    # r_d rises to a fold near r = 0.905. The guard at radial = 0.8 cuts in near r = 0.917,
    # just past it, so the valid part may dip by a sliver but never folds back.
    r = np.linspace(0.0, 1.5, 3001)
    angle = np.radians(angle_deg)
    proj = project_camera_points(np.stack([np.ones_like(r), -r * np.cos(angle), -r * np.sin(angle)], axis=1), WAYMO)
    f_u, f_v, c_u, c_v = WAYMO.intrinsic[:4]
    r_d = np.hypot((proj.uv[:, 0] - c_u) / f_u, (proj.uv[:, 1] - c_v) / f_v)

    kept = r_d[proj.valid]
    assert proj.valid[:kept.size].all()  # one unbroken run out from the axis
    assert r[kept.size - 1] == pytest.approx(0.917, abs=0.005)
    assert (np.maximum.accumulate(kept) - kept).max() < 1e-3
    assert (np.maximum.accumulate(r_d) - r_d).max() > 0.5  # unguarded, it folds all the way back


def test_box_corners_dimensions():
    c = box_corners([5, 2, 1], [4.0, 2.0, 1.5], np.pi / 2)
    np.testing.assert_allclose(c.mean(axis=0), [5, 2, 1], atol=1e-12)
    np.testing.assert_allclose(c[0] - c[4], [0, 4.0, 0], atol=1e-12)  # length now along +y
    np.testing.assert_allclose(np.linalg.norm(c[0] - c[1]), 2.0)
    np.testing.assert_allclose(c[0, 2] - c[3, 2], 1.5)


def test_box_edges_follow_the_box():
    c = box_corners([0, 0, 0], [4.0, 2.0, 1.5], 0.3)
    lengths = sorted(round(float(np.linalg.norm(c[a] - c[b])), 9) for a, b in BOX_EDGES)
    assert lengths == [1.5] * 4 + [2.0] * 4 + [4.0] * 4  # no diagonals


def test_box_rectangle_covers_the_corners_unclamped():
    # Hangs off the left of the image: comparing against an oracle needs the true edge.
    corners = box_corners([5.0, 3.0, 0.0], [1.0, 2.0, 1.5], 0.0)
    u_min, u_max, v_min, v_max = box_rectangle(corners, WAYMO)
    uv = project_camera_points(corners, WAYMO).uv
    assert (u_min, u_max, v_min, v_max) == (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())
    assert u_min < 0 < u_max


@pytest.mark.parametrize("center_x, length", [(1.0, 4.0), (0.0, 4.0), (-5.0, 4.0)])
def test_box_rectangle_is_none_with_a_corner_behind_the_camera(center_x, length):
    assert box_rectangle(box_corners([center_x, 0.0, 0.0], [length, 2.0, 1.5], 0.0), WAYMO) is None


def test_wireframe_keeps_a_half_visible_box():
    # The back face is behind the camera, and the front face is in plain view.
    corners = box_corners([1.0, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0)
    assert box_rectangle(corners, WAYMO) is None

    pieces = wireframe(corners, WAYMO)
    lengths = sorted(len(piece) for piece in pieces)
    assert lengths[-4:] == [12, 12, 12, 12]  # the front face's four edges survive whole
    assert 0 < lengths[0] < 12               # the side edges stop where the lens model does
    assert inside(np.vstack(pieces), WAYMO).any()
    assert np.isfinite(np.vstack(pieces)).all()


def test_wireframe_of_a_visible_box_has_every_edge():
    pieces = wireframe(box_corners([20.0, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0), WAYMO)
    assert len(pieces) == len(BOX_EDGES) and all(len(piece) == 12 for piece in pieces)


def test_wireframe_edges_curve_with_the_lens():
    # A post near the image edge. Two projected ends and a straight line would miss it by
    # tens of pixels, which is why edges are sampled in 3D.
    piece = wireframe(box_corners([5.0, 3.0, 0.0], [0.2, 0.2, 5.0], 0.0), WAYMO)[1]
    chord = (piece[0] + piece[-1]) / 2.0
    assert np.linalg.norm((piece[5] + piece[6]) / 2.0 - chord) > 20.0


def test_true_runs():
    assert true_runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    assert true_runs(np.array([True, True])) == [(0, 2)]
    assert true_runs(np.array([False, False])) == []

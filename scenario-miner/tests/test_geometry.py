import numpy as np
import pytest

from scenario_miner.geometry import (
    CameraModel, box_corners, densify, distance_to_segments, inv_se3, points_in_polygon,
    polyline_segments, project_points, project_polyline, rot_y, rot_z, se3, transform_points,
)

PINHOLE = CameraModel((1000.0, 1000.0, 960.0, 640.0, 0, 0, 0, 0, 0), np.eye(4), 1920, 1280)


def test_se3_inverse_roundtrip():
    T = se3(rot_z(0.7) @ rot_y(-0.1), [3.0, -2.0, 1.5])
    pts = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(transform_points(inv_se3(T), transform_points(T, pts)), pts, atol=1e-12)


def test_waymo_camera_axes():
    # Camera frame is x forward, y left, z up (not OpenCV's x right, y down, z forward).
    proj = project_points([[10, 0, 0], [10, 1, 0], [10, 0, 1], [-10, 0, 0]], PINHOLE)
    np.testing.assert_allclose(proj.uv[0], [960, 640])
    assert proj.uv[1, 0] < 960          # left in the world -> left in the image
    assert proj.uv[2, 1] < 640          # up in the world -> up in the image
    np.testing.assert_allclose(proj.uv[1], [860, 640])  # f * y / x = 100 px
    assert proj.in_image[:3].all() and not proj.in_image[3] and not proj.in_fov[3]


def test_extrinsic_is_camera_to_vehicle():
    cam = CameraModel(PINHOLE.intrinsic, se3(np.eye(3), [1.5, 0.0, 2.0]), 1920, 1280)
    # A point at the camera's height straight ahead lands on the principal point.
    np.testing.assert_allclose(project_points([[11.5, 0.0, 2.0]], cam).uv[0], [960, 640])


def test_distortion_folding_is_rejected():
    # Strong barrel distortion: far off-axis points would fold back into the image.
    cam = CameraModel((1000.0, 1000.0, 960.0, 640.0, -0.6, 0.0, 0, 0, 0), np.eye(4), 1920, 1280)
    far_left = project_points([[1.0, 1.4, 0.0]], cam)  # 54 degrees off-axis
    u_n = -1.4
    r = 1 + -0.6 * u_n**2
    assert 0 <= 1000 * u_n * r + 960 < 1920   # naive projection would claim it's visible...
    assert not far_left.in_image[0]           # ...but we reject it


def test_box_corners_dimensions():
    c = box_corners([5, 2, 1], [4.0, 2.0, 1.5], np.pi / 2)
    np.testing.assert_allclose(c.mean(axis=0), [5, 2, 1], atol=1e-12)
    np.testing.assert_allclose(c[0] - c[4], [0, 4.0, 0], atol=1e-12)  # length now along +y
    np.testing.assert_allclose(np.linalg.norm(c[0] - c[1]), 2.0)
    np.testing.assert_allclose(c[0, 2] - c[3, 2], 1.5)


def test_densify_max_spacing():
    pts = densify(np.array([[0, 0, 0], [10, 0, 0], [10, 3, 0]], dtype=float), 1.0)
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() <= 1.0 + 1e-9
    np.testing.assert_allclose(pts[[0, -1]], [[0, 0, 0], [10, 3, 0]])


def test_polyline_cut_behind_camera():
    line = np.array([[-20.0, -2.0, 0.0], [40.0, -2.0, 0.0]])
    pieces = project_polyline(line, PINHOLE)
    assert len(pieces) == 1 and len(pieces[0]) > 10
    assert np.isfinite(pieces[0]).all()


def test_distance_and_polygon():
    a, b = polyline_segments([np.array([[0, 0, 0], [10, 0, 0]]), np.array([[0, 5, 0], [10, 5, 0]])])
    d = distance_to_segments(np.array([[5, 1, 0], [5, 4.5, 0], [-3, 0, 0]]), a, b)
    np.testing.assert_allclose(d, [1.0, 0.5, 3.0])
    assert np.isinf(distance_to_segments(np.zeros((1, 3)), np.empty((0, 2)), np.empty((0, 2)))).all()
    square = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=float)
    assert points_in_polygon(np.array([[2, 2, 0], [5, 2, 0], [-1, -1, 0]]), square).tolist() == [True, False, False]


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_projection_invariant_to_rigid_motion(yaw):
    """Moving both the camera mount and the points by the same transform changes nothing."""
    T = se3(rot_z(yaw), [4.0, -1.0, 0.3])
    pts = np.array([[12.0, 1.0, 0.5], [30.0, -4.0, 1.0]])
    moved_cam = CameraModel(PINHOLE.intrinsic, T @ PINHOLE.extrinsic, 1920, 1280)
    np.testing.assert_allclose(project_points(transform_points(T, pts), moved_cam).uv,
                               project_points(pts, PINHOLE).uv, atol=1e-9)

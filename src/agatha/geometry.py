"""SE(3) transforms, Waymo camera projection, and box wireframes (NumPy).

Frames, following Waymo Open Dataset conventions:
  world   -- East-North-Up with a remote origin, so only meaningful within a segment.
  vehicle -- x forward, y left, z up; origin on the ground at the rear axle.
  camera  -- x forward along the optical axis, y left, z up. This is NOT the
             OpenCV optical frame (x right, y down, z forward), which is why the
             projection below uses u = -y/x and v = -z/x, and depth is x.

Extrinsics map camera -> vehicle. Poses map vehicle -> world.

Two different vehicle frames matter. 3D labels live in the frame pose
(vehicle_pose, near mid-spin of the top LiDAR). Each image has its own pose near
its capture time. `camera_from_label` bridges them. At 25 m/s, using the frame
pose for FRONT_LEFT or FRONT_RIGHT is about 30 px off at 15 m.

All of this needs float64: epoch timestamps collapse in float32, and world
coordinates lose centimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def se3(rotation: np.ndarray, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def transform_points(T: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap_angle(angle):
    """To [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def transform_from_list(values) -> np.ndarray:
    """4x4 float64 matrix from a row-major Parquet list<double>[16]."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# Corner order: 0-3 are the front face (+x), 4-7 the back face; each face is
# top-left, top-right, bottom-right, bottom-left when viewed from outside.
_CORNER_SIGNS = np.array(
    [
        [1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1],
        [-1, 1, 1], [-1, -1, 1], [-1, -1, -1], [-1, 1, -1],
    ],
    dtype=np.float64,
)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def box_corners(center, size, heading: float) -> np.ndarray:
    """8x3 corners of a box with size (length, width, height) and yaw `heading`."""
    half = 0.5 * np.asarray(size, dtype=np.float64)
    return (_CORNER_SIGNS * half) @ rot_z(heading).T + np.asarray(center, dtype=np.float64)


def row_corners(row, prefix: str = "") -> np.ndarray:
    """A tracks row's 8 corners, from its label columns or, with prefix "synced_", its synced ones."""
    return box_corners([getattr(row, f"{prefix}center_{axis}_m") for axis in "xyz"],
                       [getattr(row, f"{prefix}size_{axis}_m") for axis in "xyz"],
                       getattr(row, f"{prefix}heading_rad"))


@dataclass(frozen=True, eq=False)
class CameraModel:
    """Waymo camera calibration.

    intrinsic: f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 (OpenCV distortion model).
    extrinsic: 4x4 camera -> vehicle transform.
    """

    intrinsic: tuple[float, ...]
    extrinsic: np.ndarray
    width: int
    height: int
    rolling_shutter_direction: int = 0

    @property
    def camera_from_vehicle(self) -> np.ndarray:
        return inv_se3(self.extrinsic)


def camera_from_label(world_from_vehicle: np.ndarray, world_from_image_vehicle: np.ndarray,
                      cam: CameraModel) -> np.ndarray:
    """Camera <- label frame, through the world, using this image's own pose."""
    return inv_se3(world_from_image_vehicle @ cam.extrinsic) @ world_from_vehicle


class Projection(NamedTuple):
    uv: np.ndarray     # (N, 2) pixel coordinates, meaningful only where valid
    valid: np.ndarray  # (N,) in front of the camera, and the lens model holds there


def project_camera_points(points_camera, cam: CameraModel, near: float = 0.1) -> Projection:
    """Project camera-frame points into the image (no rolling-shutter correction).

    Images keep their lens distortion, so it is applied forward here. Waymo's
    guard on the radial factor is what stops the polynomial folding off-axis
    points back into view: a car beside the ego would otherwise draw a ghost box
    inside the image. cv2.projectPoints has no such guard.
    """
    p = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    depth = p[:, 0]
    in_front = depth > near
    safe = np.where(in_front, depth, 1.0)
    un, vn = -p[:, 1] / safe, -p[:, 2] / safe

    f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = cam.intrinsic
    r2 = un * un + vn * vn
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    valid = in_front & (radial >= 0.8) & (radial <= 1.2)

    ud = un * radial + 2.0 * p1 * un * vn + p2 * (r2 + 2.0 * un * un)
    vd = vn * radial + p1 * (r2 + 2.0 * vn * vn) + 2.0 * p2 * un * vn
    return Projection(np.stack([f_u * ud + c_u, f_v * vd + c_v], axis=1), valid)


def box_rectangle(corners_camera: np.ndarray, cam: CameraModel) -> tuple[float, float, float, float] | None:
    """(u_min, u_max, v_min, v_max) of the 8 corners, or None if any corner is invalid.

    Deliberately not clamped to the image, so it can be compared edge by edge
    against an unclamped projected_lidar_box row.
    """
    proj = project_camera_points(corners_camera, cam)
    if not proj.valid.all():
        return None
    return (float(proj.uv[:, 0].min()), float(proj.uv[:, 0].max()),
            float(proj.uv[:, 1].min()), float(proj.uv[:, 1].max()))


def wireframe(corners_camera: np.ndarray, cam: CameraModel, samples_per_edge: int = 12) -> list[np.ndarray]:
    """2D polylines for a box's 12 edges.

    Edges are sampled in 3D because a straight edge curves under distortion, and
    because one corner failing shouldn't lose the whole box: a car half out of
    frame still draws. Each piece runs while the lens model holds. Drawing clips
    to the image, so pieces may extend past it. Projection ignores occlusion, so
    a drawn box doesn't mean the object is visible.
    """
    t = np.linspace(0.0, 1.0, samples_per_edge)[:, None]
    pieces = []
    for a, b in BOX_EDGES:
        points = corners_camera[a] * (1.0 - t) + corners_camera[b] * t
        proj = project_camera_points(points, cam)
        pieces += [proj.uv[s:e] for s, e in true_runs(proj.valid) if e - s >= 2]
    return pieces

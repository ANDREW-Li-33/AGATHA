"""SE(3) transforms, Waymo camera projection, and 2D map geometry (NumPy).

Frames, following Waymo Open Dataset conventions:
  world   -- global frame the HD map lives in (after adding map_pose_offset).
  vehicle -- x forward, y left, z up; origin at the rear axle.
  camera  -- x forward along the optical axis, y left, z up. This is NOT the
             OpenCV optical frame (x right, y down, z forward), which is why the
             projection below uses u = -y/x and v = -z/x.

Calibration extrinsics map camera -> vehicle. Poses map vehicle -> world.
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


def rot_y(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


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

    @property
    def camera_from_vehicle(self) -> np.ndarray:
        return inv_se3(self.extrinsic)

    def to_dict(self) -> dict:
        return {
            "intrinsic": [float(v) for v in self.intrinsic],
            "extrinsic": np.asarray(self.extrinsic).reshape(-1).tolist(),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(
            intrinsic=tuple(d["intrinsic"]),
            extrinsic=np.asarray(d["extrinsic"], dtype=np.float64).reshape(4, 4),
            width=int(d["width"]),
            height=int(d["height"]),
        )


class Projection(NamedTuple):
    uv: np.ndarray        # (N, 2) pixel coordinates
    depth: np.ndarray     # (N,) distance along the optical axis, meters
    in_image: np.ndarray  # (N,) in front of the camera and inside the image
    in_fov: np.ndarray    # (N,) in front of the camera and inside a padded FOV


def project_points(points_vehicle, cam: CameraModel, near: float = 0.1,
                   fov_margin: float = 1.15) -> Projection:
    """Project vehicle-frame points into the image (no rolling-shutter correction)."""
    p = transform_points(cam.camera_from_vehicle, points_vehicle)
    depth = p[:, 0]
    in_front = depth > near
    safe = np.where(in_front, depth, 1.0)
    un, vn = -p[:, 1] / safe, -p[:, 2] / safe

    f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = cam.intrinsic
    # The distortion polynomial is only valid near the image; far outside it the
    # radial term turns over and folds points back into view. Gate on a padded
    # undistorted FOV before trusting the distorted coordinates.
    lim_u = fov_margin * max(c_u, cam.width - c_u) / f_u
    lim_v = fov_margin * max(c_v, cam.height - c_v) / f_v
    in_fov = in_front & (np.abs(un) <= lim_u) & (np.abs(vn) <= lim_v)

    r2 = un * un + vn * vn
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    ud = un * radial + 2.0 * p1 * un * vn + p2 * (r2 + 2.0 * un * un)
    vd = vn * radial + p1 * (r2 + 2.0 * vn * vn) + 2.0 * p2 * un * vn
    uv = np.stack([f_u * ud + c_u, f_v * vd + c_v], axis=1)

    in_image = (
        in_fov
        & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    )
    return Projection(uv, depth, in_image, in_fov)


def densify(points: np.ndarray, max_spacing: float) -> np.ndarray:
    """Insert points so no polyline segment is longer than `max_spacing`."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    seg = np.diff(points, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / max_spacing).astype(int))
    idx = np.repeat(np.arange(len(seg)), n)
    step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = step / np.repeat(n, n)
    return np.vstack([points[idx] + seg[idx] * t[:, None], points[-1:]])


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def project_polyline(points_vehicle, cam: CameraModel, closed: bool = False,
                     max_spacing: float = 1.0) -> list[np.ndarray]:
    """Project a 3D polyline into the image as a list of 2D pieces.

    The line is densified first, so cutting it where it leaves the FOV (or goes
    behind the camera) loses at most `max_spacing` meters at each cut.
    """
    pts = np.asarray(points_vehicle, dtype=np.float64)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    pts = densify(pts, max_spacing)
    proj = project_points(pts, cam)
    return [proj.uv[a:b] for a, b in true_runs(proj.in_fov) if b - a >= 2]


def polyline_segments(polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack the 2D segments of many polylines without joining polyline ends."""
    starts, ends = [], []
    for line in polylines:
        line = np.asarray(line, dtype=np.float64)[:, :2]
        if len(line) == 1:
            line = np.vstack([line, line])
        starts.append(line[:-1])
        ends.append(line[1:])
    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(starts), np.vstack(ends)


def distance_to_segments(points, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Min 2D distance from each point to any segment (inf if there are none)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, np.shape(points)[-1])[:, None, :2]
    if len(seg_a) == 0:
        return np.full(P.shape[0], np.inf)
    A, AB = seg_a[None], (seg_b - seg_a)[None]
    denom = np.maximum((AB * AB).sum(-1), 1e-12)
    t = np.clip(((P - A) * AB).sum(-1) / denom, 0.0, 1.0)
    closest = A + t[..., None] * AB
    return np.linalg.norm(P - closest, axis=-1).min(axis=1)


def points_in_polygon(points, polygon) -> np.ndarray:
    """Even-odd rule point-in-polygon test in the xy plane."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, np.shape(points)[-1])[:, :2]
    poly = np.asarray(polygon, dtype=np.float64)[:, :2]
    x, y = P[:, 0:1], P[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (crosses & (x < x_at_y)).sum(axis=1) % 2 == 1

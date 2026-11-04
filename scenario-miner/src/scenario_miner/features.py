"""Per-object 3D features: range, camera visibility, and position relative to the HD map.

These feed three consumers: the `objects` table (and from it the SQL-derived
ground truth), the geometric baseline, and the text scene summary given to the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import box_corners, distance_to_segments, points_in_polygon, polyline_segments, project_points
from .sample import Sample

MAX_VISIBLE_RANGE_M = 60.0


@dataclass
class ObjectFeatures:
    object_id: str
    type: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    heading: float
    speed: float
    range_m: float
    lane_dist_m: float | None   # distance to the nearest drivable lane centerline
    in_crosswalk: bool
    in_camera: bool             # some part of the box projects into this camera's image
    num_lidar_points: int
    attributes: dict

    @property
    def on_roadway(self) -> bool:
        return self.in_crosswalk or (self.lane_dist_m is not None and self.lane_dist_m <= 1.5)


def object_features(sample: Sample, max_range: float = MAX_VISIBLE_RANGE_M) -> list[ObjectFeatures]:
    if not sample.boxes:
        return []
    centers = np.stack([b.center for b in sample.boxes])
    ranges = np.linalg.norm(centers[:, :2], axis=1)

    seg_a, seg_b = polyline_segments(sample.lanes())
    lane_dist = distance_to_segments(centers, seg_a, seg_b)

    in_crosswalk = np.zeros(len(centers), dtype=bool)
    for f in sample.map_features:
        if f.kind == "crosswalk" and len(f.points) >= 3:
            in_crosswalk |= points_in_polygon(centers, f.points)

    # Visible = center or any corner lands in the image. A label that the camera
    # can't see must not count as ground truth for that camera's frame.
    pts = np.concatenate([centers[:, None, :], np.stack([box_corners(b.center, b.size, b.heading)
                                                          for b in sample.boxes])], axis=1)
    proj = project_points(pts.reshape(-1, 3), sample.camera_model)
    any_in_image = proj.in_image.reshape(len(centers), 9).any(axis=1)

    out = []
    for i, b in enumerate(sample.boxes):
        visible = bool(any_in_image[i] and ranges[i] <= max_range and b.num_lidar_points != 0)
        out.append(ObjectFeatures(
            object_id=b.object_id,
            type=b.type,
            x=float(b.center[0]), y=float(b.center[1]), z=float(b.center[2]),
            length=float(b.size[0]), width=float(b.size[1]), height=float(b.size[2]),
            heading=float(b.heading),
            speed=float(b.speed),
            range_m=float(ranges[i]),
            lane_dist_m=None if not np.isfinite(lane_dist[i]) else float(lane_dist[i]),
            in_crosswalk=bool(in_crosswalk[i]),
            in_camera=visible,
            num_lidar_points=int(b.num_lidar_points),
            attributes=b.attributes,
        ))
    return out

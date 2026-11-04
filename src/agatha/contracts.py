"""SceneWindow: the Stage 1 -> Stage 2 contract.

Three named coordinate frames, and field names carry frame and unit:
  world          -- float64 East-North-Up with a remote origin. tracks.parquet only.
  segment_local  -- world minus the first frame's translation, axes still ENU, so
                    Waymo's speed.x and speed.y stay valid in it.
  ego_at_sample  -- forward-left-up at that sample's vehicle_pose, origin on the
                    ground at the rear axle. x overstates clearance by the distance
                    to the front bumper; see EgoFootprint.

Times: frame_timestamp_micros is the exact key, and t_s is seconds since the
segment's first frame, for reading only. Never join or compare on t_s.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "0.1"
SOURCE_DATASET_ATTRIBUTION = (
    "This work was made using the Waymo Open Dataset, provided by Waymo LLC under the "
    "Waymo Dataset License Agreement for Non-Commercial Use, available at "
    "waymo.com/open/terms, and your access and use of such work are governed by the "
    "terms and conditions therein."
)

VelocitySource = Literal["label", "central_difference", "none"]


class _Contract(BaseModel):
    """NaN would be written as null, which a plain float field refuses on read. So it fails when it's built."""

    model_config = ConfigDict(allow_inf_nan=False)


class FrameRef(_Contract):
    frame_timestamp_micros: int
    t_s: float
    camera: str
    capture_offset_ms: float
    clean_ref: str       # original JPEG bytes, relative to the segment's output directory
    annotated_ref: str   # wireframes for this window's render profile


class ObjectSample(_Contract):
    frame_timestamp_micros: int
    t_s: float
    type: str                       # per sample: pedestrian and cyclist can switch
    pos_ego_x_m: float              # box center, ego_at_sample
    pos_ego_y_m: float
    front_gap_m: float | None       # box corners to ego front bumper, if ahead of it
    along_path_m: float             # along the ego's post hoc path, relative to ego now
    lateral_offset_m: float         # box center from the ego path, left positive
    on_driven_path: bool            # False past the path's ends: the two above then use a straight line, not a road
    path_clearance_m: float | None  # nearest box corner to the ego path, 0 if straddling, None on another level
    v_abs_ego_x_mps: float | None   # over the ground, in ego axes
    v_abs_ego_y_mps: float | None
    v_rel_ego_x_mps: float | None   # relative to the ego, in ego axes
    v_rel_ego_y_mps: float | None
    a_abs_ego_x_mps2: float | None
    a_abs_ego_y_mps2: float | None
    v_source: VelocitySource
    fov_cameras: list[str]          # has a projected_lidar_box row: in frame, not necessarily visible
    not_directly_observed: bool     # no top-LiDAR points in the box
    stationary: bool
    context_only: bool              # outside the image window: track context, no frame shown


class ObjectTrack(_Contract):
    alias: str               # stable per segment: drawn on images, used in prompts
    laser_object_id: str
    type: str                # type at first appearance
    first_t_s: float
    last_t_s: float
    range_at_first_seen_m: float
    truncated_start: bool    # already present in the segment's first frame
    truncated_end: bool      # still present in the segment's last frame
    samples: list[ObjectSample]


class OmittedObject(_Contract):
    """Past max_objects, so no track here, but the images still label it, so its alias is a valid citation."""

    alias: str
    laser_object_id: str
    type: str                # type at first appearance


class EgoSample(_Contract):
    frame_timestamp_micros: int
    t_s: float
    pos_local_x_m: float
    pos_local_y_m: float
    speed_mps: float | None
    a_long_mps2: float | None
    a_lat_mps2: float | None
    yaw_rate_radps: float | None


class EgoFootprint(_Contract):
    rear_axle_to_front_m: float
    assumption: str = "Configured estimate, not a Waymo figure: Perception has no ego box."


class SceneLimits(_Contract):
    """What the labels can't say. Carried into every Stage 2 prompt."""

    labeled_radius_m: float = 75.0
    camera_coverage_deg: float = 230.0
    absent_box_is_not_clear_space: str = (
        "No box does not mean no object: labels stop at the radius above, No Label Zones "
        "are unlabeled, and cones, barriers, and debris have no 3D boxes."
    )
    abrupt_track_start: str = (
        "A track that starts abruptly may be a labeling limit rather than an object appearing."
    )
    no_vehicle_performance_claims: str = (
        "Manual and autonomous driving are mixed without labels, so nothing here supports a "
        "conclusion about a real vehicle's behavior or performance."
    )


class SourceRef(_Contract):
    """A logical reference. The root comes from config, so these survive host changes."""

    component: str
    split: str
    context_name: str
    dataset_version: str


class SceneWindow(_Contract):
    schema_version: str = SCHEMA_VERSION
    segment_context_name: str
    split: str
    t0_micros: int
    window_start_s: float    # the window bounds grid times, not frames: jitter or force_include can put a
    window_end_s: float      # frame outside them, so validate a cited frame against `frames`
    cameras: list[str]
    profile: str
    requested_fps: float
    achieved_fps: float
    max_gap_ms: float
    frames: list[FrameRef]
    objects: list[ObjectTrack]
    omitted_objects: list[OmittedObject]
    selection_rule: str
    ego: list[EgoSample]
    ego_footprint: EgoFootprint
    ego_path: str = "post_hoc: the ego's own driven path, extended straight along its first and last headings"
    limits: SceneLimits = SceneLimits()
    sources: list[SourceRef]
    preprocessing_version: str
    attribution: str = SOURCE_DATASET_ATTRIBUTION

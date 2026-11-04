"""What a model reads: two system prompts, with all that never changes, and content built from a SceneWindow.

Content is the window's annotated images, each after a caption, then its tracks as compact JSON, naming no segment."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import ObjectTrack, SceneWindow
from .schema import TAXONOMY, Candidate

# Bump on any change to what a request says. Stored answers key on the text, so a missed bump costs only provenance.
PROMPT_VERSION = "2026-11-04.2"

_COMMON = """You review camera frames and tracked road users from a vehicle's sensor log, for a research dataset of
driving incidents.

Rule. An event is relevant if unexpected road-user behavior poses a potential hazard, or if construction or an
obstruction materially changes the expected driving space, route, or traffic control. Mere visibility of roadside
construction does not qualify. A detection is not proof that the recorded vehicle failed.

Taxonomy, as label (category): definition. An event can take several labels, each with its own evidence. Prefer
what is directly observable over speculation, and give each event its own start and end.
""" + "\n".join(f"- {label} ({category}): {text}" for label, (category, text) in TAXONOMY.items()) + """

Inputs. Frames from the recording vehicle's (the ego's) cameras, each after a caption with its camera and t_s,
seconds since the segment's first frame. Wireframe boxes mark labeled road users, tagged with aliases such as V003
(vehicle), P001 (pedestrian) or C001 (cyclist); a gray box had no LiDAR points then, so it may be occluded. Then JSON:
- window: the span the images cover, start_s to end_s, with its frame rate and the cameras shown, not all the ego's.
- objects: tracked road users. seen_s is the track's whole span in the segment. Rows follow object_columns: x_m and
  y_m are the box center in the ego's frame at that time, x forward and y left, in meters from the rear axle;
  lateral_m is the center's offset from the ego's own driven path, the stand-in for its lane, left positive;
  front_gap_m runs from the ego's front bumper to the box when it's ahead, else null; v_rel_* is velocity relative
  to the ego, in the ego's axes. When given, path_clearance_m is the box's nearest corner to the ego's path, 0 when
  it straddles it, and a_x_mps2 and a_y_mps2 are acceleration over the ground, in the ego's axes. Rows outside the
  window are track context, with no image. Flags: stationary; present_at_segment_start and present_at_segment_end,
  when the recording cut the track; not_directly_observed_at_times; past_driven_path_at_times, when lateral_m is
  from a straight extension of the path; type_changes.
- omitted_objects: road users the images label but the JSON doesn't track. Their aliases are valid.
- ego: the ego's speed, longitudinal and lateral acceleration, and yaw rate, in rows that follow ego_columns.
- limits: what the labels can't show. Cones, barriers and debris have no boxes, so look for them in the images.
"""

PASS1_SYSTEM = _COMMON + """
Task. Find every candidate event in this window that might fit the taxonomy. Recall matters more than precision
here: a second pass verifies each candidate with more frames, so include the plausible ones and leave the doubt to
it. Return no candidates only when nothing in the window could qualify. For each, give its labels; start_s and
end_s, its own span inside the window; the aliases of the objects involved, if any; the visual and motion
evidence, with times; and driving_impact, a short note on how it could affect the ego's driving."""

PASS2_SYSTEM = _COMMON + """
Task. Verify one candidate event that a first pass found at a lower frame rate. Test the hypothesis rather than
assume it: for each label, give the evidence for it and the evidence against it, then its status: CONFIRMED when
the frames and tracks show it, REJECTED when they show it didn't happen or doesn't meet the rule, UNCERTAIN when
they can't settle it, with the reasons in uncertainty_reasons. Assess every hypothesized label, and add any other
label the evidence supports. The event's status is its labels' strongest: CONFIRMED if any label is, else
UNCERTAIN if any is, else REJECTED. Give the event's own start_s and end_s inside the window, the aliases
involved, and source_frames: the frames that show the evidence, by camera and t_s as captioned, at least one for
a CONFIRMED event. explanation: a concise, evidence-based account of what the frames and tracks show, for a human
reviewer. driving_impact: how the event could affect the ego's driving."""

# JSON column -> ObjectSample field. Pass 2 adds what verification needs, for braking and the moment of entry.
_PASS1_COLUMNS = {"t_s": "t_s", "x_m": "pos_ego_x_m", "y_m": "pos_ego_y_m", "lateral_m": "lateral_offset_m",
                  "front_gap_m": "front_gap_m", "v_rel_x_mps": "v_rel_ego_x_mps", "v_rel_y_mps": "v_rel_ego_y_mps"}
_PASS2_COLUMNS = {**_PASS1_COLUMNS, "path_clearance_m": "path_clearance_m", "a_x_mps2": "a_abs_ego_x_mps2",
                  "a_y_mps2": "a_abs_ego_y_mps2"}
_EGO_COLUMNS = ("t_s", "speed_mps", "a_long_mps2", "a_lat_mps2", "yaw_rate_radps")


def _number(value):
    """Two decimals: centimeters and hundredths of a second are finer than the labels support."""
    return round(value, 2) if isinstance(value, float) else value


def _flags(track: ObjectTrack) -> list[str]:
    samples = track.samples
    flags = {
        "stationary": bool(samples) and all(sample.stationary for sample in samples),
        "present_at_segment_start": track.truncated_start,
        "present_at_segment_end": track.truncated_end,
        "not_directly_observed_at_times": any(sample.not_directly_observed for sample in samples),
        "past_driven_path_at_times": not all(sample.on_driven_path for sample in samples),
        "type_changes": len({sample.type for sample in samples}) > 1,
    }
    return [name for name, on in flags.items() if on]


def window_context(window: SceneWindow, columns: dict[str, str]) -> dict:
    """The window's tracks as compact JSON: column names once, then rows of numbers."""
    return {
        "window": {"start_s": _number(window.window_start_s), "end_s": _number(window.window_end_s),
                   "cameras": window.cameras, "fps": _number(window.requested_fps)},
        "object_columns": list(columns),
        "objects": [{"alias": track.alias, "type": track.type, "flags": _flags(track),
                     "seen_s": [_number(track.first_t_s), _number(track.last_t_s)],
                     "rows": [[_number(getattr(sample, field)) for field in columns.values()]
                              for sample in track.samples]}
                    for track in window.objects],
        "omitted_objects": [{"alias": item.alias, "type": item.type} for item in window.omitted_objects],
        "ego_columns": list(_EGO_COLUMNS),
        "ego": [[_number(getattr(sample, name)) for name in _EGO_COLUMNS] for sample in window.ego],
        "limits": window.limits.model_dump(),
    }


def request_content(window: SceneWindow, directory: Path, hypothesis: Candidate | None = None) -> list[dict]:
    """Pass 1's content, or with a hypothesis Pass 2's: richer rows, then the hypothesis to test."""
    content = []  # each annotated frame after its caption, with a ref relative to the segment's directory
    for frame in window.frames:
        image = {"type": "image", "ref": frame.annotated_ref, "data": (directory / frame.annotated_ref).read_bytes()}
        content += [{"type": "text", "text": f"{frame.camera} t={frame.t_s:.2f} s"}, image]
    context = window_context(window, _PASS1_COLUMNS if hypothesis is None else _PASS2_COLUMNS)
    content.append({"type": "text", "text": json.dumps(context, separators=(",", ":"))})
    if hypothesis is not None:
        content.append({"type": "text", "text": "Hypothesis from the first pass: " + hypothesis.model_dump_json()})
    return content

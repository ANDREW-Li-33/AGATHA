"""Wireframe overlays, the render cache, clips, and the QA contact sheet.

Overlays are visual aids, not measurements. A box comes from lidar_camera_synced_box when
this camera is the object's most visible one, otherwise from lidar_box through the image's
own pose, and nothing is extrapolated in time. Clean images stay the original JPEG bytes,
because cones, barriers, and debris have no boxes, so a wireframe can hide the evidence.

Files are named by timestamp and profile, so windows and rates share them. That's why every
road user in a frame is drawn, not just the objects a window kept: what's in a shared file
can't depend on which window asked first.
"""

from __future__ import annotations

import base64
import html
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from .columns import CAMERA_IDS, CAMERA_NAMES, TIME
from .config import RenderProfile
from .contracts import SceneWindow
from .files import write_atomic
from .geometry import CameraModel, camera_from_label, row_corners, transform_from_list, transform_points, wireframe
from .loader import SegmentRef, read_jpegs
from .store import SegmentOutputs, camera_models

# BGR, as OpenCV wants it. Types not listed, sign and unknown, aren't drawn.
_BOX_BGR = {"vehicle": (255, 150, 0), "pedestrian": (50, 50, 255), "cyclist": (0, 220, 255)}
DRAWN_TYPES = frozenset(_BOX_BGR)
_OCCLUDED_BGR = (170, 170, 170)
_LABEL_BACKGROUND_BGR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_THICK = 3  # times the profile's line width, for the one box a reviewer is asked about

_LAST_FRAME_S = 0.1  # it has no next timestamp, so it gets the cameras' nominal 10 Hz period
_CLIP_TIMEOUT_S = 300  # a full-resolution 20 s clip at 10 FPS takes well under this

_CONTACT_SHEET_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Contact sheet</title>
<style>
  body { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 16px;
         margin: 16px; font-family: sans-serif; }
  figure { margin: 0; }
  img { width: 100%; }
</style>
"""


class DrawBox(NamedTuple):
    alias: str
    type: str
    corners_camera: np.ndarray  # (8, 3)
    draw_source: str            # "camera_synced_box" or "lidar_box"
    occluded: bool              # not_directly_observed


class ContactItem(NamedTuple):
    jpeg: bytes
    caption: str


def _in_image(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    return ((uv >= 0.0) & (uv < (cam.width, cam.height))).all(axis=1)


def uses_synced(row, camera_id: int) -> bool:
    """Whether this camera draws a tracks row from its synced box. A NaN in any value reaches the corners."""
    return row.synced_camera == camera_id and bool(np.isfinite(row_corners(row, "synced_")).all())


def draw_boxes(tracks_at_time: pd.DataFrame, camera_id: int, cam: CameraModel,
               camera_from_label_frame: np.ndarray) -> list[DrawBox]:
    """The boxes one image shows. Any with no wireframe point inside the image are left out.

    Waymo projects a synced box with the extrinsic alone, because it has already been moved to
    this camera's capture time. Sending it through the image pose would move it a second time.
    Waymo says some synced values are missing, and the object's label then stands in.
    """
    boxes = []
    for row in tracks_at_time.itertuples():
        if row.type not in DRAWN_TYPES:
            continue
        if uses_synced(row, camera_id):
            source = "camera_synced_box"
            corners = transform_points(cam.camera_from_vehicle, row_corners(row, "synced_"))
        else:
            source, corners = "lidar_box", transform_points(camera_from_label_frame, row_corners(row))
        pieces = wireframe(corners, cam)
        if pieces and _in_image(np.vstack(pieces), cam).any():
            boxes.append(DrawBox(row.alias, row.type, corners, source, row.not_directly_observed))
    return boxes


def _draw_label(image: np.ndarray, text: str, top_left: np.ndarray, color, profile: RenderProfile) -> None:
    """A tag above the box, pushed back when it would fall off the top or right of the image."""
    thickness = max(profile.line_width - 1, 1)
    (width, height), baseline = cv2.getTextSize(text, _FONT, profile.font_scale, thickness)
    x = min(int(top_left[0]), image.shape[1] - width - 1)
    y = max(int(top_left[1]), height + baseline)
    cv2.rectangle(image, (x, y - height - baseline), (x + width, y), _LABEL_BACKGROUND_BGR, cv2.FILLED)
    cv2.putText(image, text, (x, y - baseline), _FONT, profile.font_scale, color, thickness, cv2.LINE_AA)


def render_annotated(jpeg: bytes, boxes, cam: CameraModel, profile: RenderProfile,
                     thick_alias: str | None = None) -> bytes:
    """Wireframes on a copy scaled to the profile, never upscaled. The box named `thick_alias` is drawn thicker.

    Scaling comes first, so that line widths and labels are the profile's size on the image
    that gets used, instead of shrinking with it. Only the cut-in review asks for a thick box:
    cached files are shared by every window, so what they show can't depend on who asked.
    """
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    scale = min(profile.max_long_edge / max(width, height), 1.0)
    image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    for box in sorted(boxes, key=lambda box: -box.corners_camera[:, 0].max()):  # keeps near labels on top
        color = _OCCLUDED_BGR if box.occluded else _BOX_BGR[box.type]
        pieces = wireframe(box.corners_camera, cam)
        line_width = profile.line_width * (_THICK if box.alias == thick_alias else 1)
        cv2.polylines(image, [np.round(piece * scale).astype(np.int32) for piece in pieces], False, color,
                      line_width, cv2.LINE_AA)
        uv = np.vstack(pieces)
        _draw_label(image, box.alias, uv[_in_image(uv, cam)].min(axis=0) * scale, color, profile)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, profile.jpeg_quality])[1].tobytes()


def boxes_for_image(segment: SegmentOutputs, timestamp: int, camera_id: int, cam: CameraModel) -> list[DrawBox]:
    """draw_boxes for one image, with its own pose. Review frames need this too: they can be in no window."""
    frames, tracks = segment.frames, segment.tracks
    frame = frames[(frames[TIME] == timestamp) & (frames["camera"] == camera_id)].iloc[0]
    camera_from_label_frame = camera_from_label(transform_from_list(frame["world_from_vehicle"]),
                                                transform_from_list(frame["world_from_image_vehicle"]), cam)
    return draw_boxes(tracks[tracks[TIME] == timestamp], camera_id, cam, camera_from_label_frame)


def ensure_rendered(window: SceneWindow, segment: SegmentOutputs, root, ref: SegmentRef,
                    profiles: dict[str, RenderProfile]) -> dict:
    """Write the window's images, skipping files that exist, and reading each camera's JPEGs once.

    The window names its profile because its refs carry that name, so pixels can't land under
    the wrong one. Returns the boxes this call drew per draw_source, so summing over overlapping
    windows counts each frame once. A call that finds everything on disk returns zeros.
    """
    directory, profile = segment.directory, profiles[window.profile]
    pending = defaultdict(list)
    for frame in window.frames:
        if not ((directory / frame.clean_ref).exists() and (directory / frame.annotated_ref).exists()):
            pending[frame.camera].append(frame)

    models = camera_models(segment.meta)
    counts = Counter(camera_synced_box=0, lidar_box=0)
    for camera, frames in pending.items():
        camera_id = CAMERA_IDS[camera]
        cam = models[camera_id]
        jpegs = read_jpegs(root, ref, camera_id, [frame.frame_timestamp_micros for frame in frames])
        for frame in frames:
            timestamp = frame.frame_timestamp_micros
            clean_path, annotated_path = directory / frame.clean_ref, directory / frame.annotated_ref
            if not annotated_path.exists():  # first: decoding here finds a damaged JPEG before it's kept as clean
                boxes = boxes_for_image(segment, timestamp, camera_id, cam)
                write_atomic(annotated_path, render_annotated(jpegs[timestamp], boxes, cam, profile))
                counts.update(box.draw_source for box in boxes)
            if not clean_path.exists():
                write_atomic(clean_path, jpegs[timestamp])
    return dict(counts)


def _concat_script(paths: list[Path], timestamps: list[int]) -> str:
    """An ffconcat script that holds each image until the next real timestamp.

    Two ffmpeg behaviors matter, both measured on the bundled build. An image's timestamp is
    rounded to 1/framerate, and the default of 25 snaps every frame to a 40 ms grid, however
    the output is set up, so `framerate` matches the microsecond timestamps. And a duration
    only counts once another file follows it, so the last image is listed twice.
    """
    lines = ["ffconcat version 1.0"]
    for path, duration_s in zip(paths, [*np.diff(timestamps) / 1e6, _LAST_FRAME_S]):
        lines += [f"file '{path}'", "option framerate 1000000", f"duration {duration_s:.6f}"]
    lines += [f"file '{paths[-1]}'", "option framerate 1000000"]
    return "\n".join(lines) + "\n"


def write_clip(segment: SegmentOutputs, camera_id: int, start_s: float, end_s: float, out_path,
               profile_name: str = "clean") -> Path:
    """H.264 clip of the frames already rendered in [start_s, end_s], at their real timing.

    A frame that was never rendered is left out, and the one before it holds until the next
    real timestamp. The ffconcat script stays beside the clip, as the record of its timing.
    """
    frames = segment.frames
    in_range = frames.loc[(frames["camera"] == camera_id) & frames["t_s"].between(start_s, end_s), TIME]
    folder = (segment.directory / "render" / CAMERA_NAMES[camera_id]).resolve()
    timestamps = [ts for ts in in_range.tolist() if (folder / f"{ts}__{profile_name}.jpg").exists()]
    if not timestamps:
        raise ValueError("no rendered frames in this range")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = out_path.with_suffix(".ffconcat")
    script.write_text(_concat_script([folder / f"{ts}__{profile_name}.jpg" for ts in timestamps], timestamps))
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(script),
        # yuv420p needs even dimensions. JPEGs are full range, which would otherwise stay as yuvj420p.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv",
        # x264 reads the 1 MHz input rate as fps and claims level 6.2. 5.0 is the lowest that fits 1920x1280.
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-level:v", "5.0",
        # Each frame keeps its own timestamp. Without this, the Linux build pads to the 1 MHz input rate.
        "-fps_mode", "passthrough",
        str(out_path),
    ]
    # Not check=True: that exception's message holds the command, and so the context name.
    # The timeout kills a stuck encode, which would otherwise spend the job's whole time limit.
    if subprocess.run(command, capture_output=True, timeout=_CLIP_TIMEOUT_S).returncode != 0:
        raise RuntimeError("ffmpeg failed")
    return out_path


def write_contact_sheet(path, items: list[ContactItem]) -> Path:
    """One HTML file with the images inside it, so it opens anywhere inside OnDemand."""
    figures = [f'<figure><img alt="" src="data:image/jpeg;base64,{base64.b64encode(item.jpeg).decode()}">'
               f"<figcaption>{html.escape(item.caption)}</figcaption></figure>\n" for item in items]
    write_atomic(path, (_CONTACT_SHEET_HEAD + "".join(figures)).encode())
    return Path(path)

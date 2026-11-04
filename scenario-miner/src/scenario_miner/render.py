"""Project HD-map features and 3D boxes onto the camera image.

Used two ways: as a visual check that calibration/pose handling is right (the
first thing to verify on real data), and as "prompt enrichment" -- the overlay
image is sent to the VLM next to the raw frame.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .features import object_features
from .geometry import BOX_EDGES, box_corners, project_points, project_polyline
from .sample import Sample

BOX_COLORS = {
    "vehicle": (0, 150, 255),
    "pedestrian": (255, 50, 50),
    "cyclist": (255, 220, 0),
    "sign": (255, 140, 0),
    "unknown": (200, 200, 200),
}
MAP_COLORS = {
    "lane": (0, 255, 255),
    "road_line": (255, 255, 255),
    "road_edge": (255, 0, 200),
    "crosswalk": (170, 80, 255),
    "speed_bump": (0, 255, 0),
    "driveway": (120, 120, 255),
    "stop_sign": (255, 0, 0),
}
TYPE_ABBREV = {"vehicle": "VEH", "pedestrian": "PED", "cyclist": "CYC", "sign": "SIGN", "unknown": "UNK"}

OVERLAY_LEGEND = (
    "Overlay legend: cyan = HD-map lane centerlines, white/yellow = painted road lines, "
    "magenta = road edges, purple = crosswalks. 3D boxes from LiDAR: blue = vehicle, "
    "red = pedestrian, yellow = cyclist, orange = sign or static object; each is tagged with its "
    "range in meters."
)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_overlay(sample: Sample, image: Image.Image | None = None, draw_map: bool = True,
                   draw_boxes: bool = True, max_range: float = 60.0) -> Image.Image:
    img = (image or sample.load_image()).convert("RGB")
    cam = sample.camera_model
    sx, sy = img.width / cam.width, img.height / cam.height  # allow drawing on resized images
    scale = np.array([sx, sy])
    width = max(2, round(img.width / 480))
    draw = ImageDraw.Draw(img)

    if draw_map:
        for f in sample.map_features:
            color = MAP_COLORS.get(f.kind, (255, 255, 255))
            if f.kind == "road_line" and 4 <= f.subtype <= 8:
                color = (255, 210, 0)  # yellow lines
            if f.kind == "stop_sign":
                proj = project_points(f.points, cam)
                if proj.in_image[0]:
                    u, v = proj.uv[0] * scale
                    r = 3 * width
                    draw.regular_polygon((u, v, r), 8, fill=color)
                continue
            for piece in project_polyline(f.points, cam, closed=f.is_polygon):
                draw.line([tuple(p) for p in piece * scale], fill=color, width=width)

    if draw_boxes:
        font = _font(max(12, img.width // 80))
        feats = {f.object_id: f for f in object_features(sample, max_range)}
        for b in sorted(sample.boxes, key=lambda b: -float(np.linalg.norm(b.center[:2]))):
            f = feats[b.object_id]
            if not f.in_camera:
                continue
            color = BOX_COLORS.get(b.type, BOX_COLORS["unknown"])
            corners = box_corners(b.center, b.size, b.heading)
            proj = project_points(corners, cam)
            uv = proj.uv * scale
            for i, j in BOX_EDGES:
                if proj.in_fov[i] and proj.in_fov[j]:
                    w = width + 1 if (i < 4 and j < 4) else width  # thicker front face shows heading
                    draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=w)
            visible = uv[proj.in_fov]
            if len(visible):
                x, y = visible[:, 0].min(), visible[:, 1].min()
                text = f"{TYPE_ABBREV.get(b.type, 'UNK')} {f.range_m:.0f}m"
                tb = draw.textbbox((x, y), text, font=font, anchor="lb")
                draw.rectangle(tb, fill=(0, 0, 0))
                draw.text((x, y), text, fill=color, font=font, anchor="lb")
    return img


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)


def to_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

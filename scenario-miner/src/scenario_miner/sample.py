"""Dataset-agnostic sample format.

Extractors (Waymo v1.4.x, synthetic, ...) write one directory per camera image:

    <root>/<sample_id>/image.jpg
    <root>/<sample_id>/sample.json

Everything in sample.json is already in the vehicle frame *at the camera's
capture time*, so downstream code never touches world poses again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .geometry import CameraModel

POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")


@dataclass
class Box3D:
    object_id: str
    type: str                 # vehicle | pedestrian | cyclist | sign | unknown
    center: np.ndarray        # (3,) vehicle frame, meters
    size: np.ndarray          # (3,) length, width, height
    heading: float            # yaw in the vehicle frame, radians
    speed: float = 0.0
    num_lidar_points: int = -1  # -1 when unknown
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.object_id,
            "type": self.type,
            "center": _round(self.center),
            "size": _round(self.size),
            "heading": round(float(self.heading), 4),
            "speed": round(float(self.speed), 3),
            "num_lidar_points": int(self.num_lidar_points),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Box3D":
        return cls(
            object_id=str(d["id"]),
            type=d["type"],
            center=np.asarray(d["center"], dtype=np.float64),
            size=np.asarray(d["size"], dtype=np.float64),
            heading=float(d["heading"]),
            speed=float(d.get("speed", 0.0)),
            num_lidar_points=int(d.get("num_lidar_points", -1)),
            attributes=d.get("attributes", {}),
        )


@dataclass
class MapFeature:
    feature_id: int
    kind: str                 # lane | road_line | road_edge | crosswalk | speed_bump | driveway | stop_sign
    points: np.ndarray        # (N, 3) vehicle frame
    subtype: int = 0          # LaneType / RoadLineType / RoadEdgeType enum value

    @property
    def is_polygon(self) -> bool:
        return self.kind in POLYGON_KINDS

    def to_dict(self) -> dict:
        return {"id": int(self.feature_id), "kind": self.kind, "subtype": int(self.subtype),
                "points": _round(self.points)}

    @classmethod
    def from_dict(cls, d: dict) -> "MapFeature":
        return cls(int(d["id"]), d["kind"], np.asarray(d["points"], dtype=np.float64).reshape(-1, 3),
                   int(d.get("subtype", 0)))


@dataclass
class Sample:
    sample_id: str
    segment_id: str
    timestamp_us: int
    camera: str
    camera_model: CameraModel
    boxes: list[Box3D]
    map_features: list[MapFeature]
    ego_speed: float | None = None
    context: dict = field(default_factory=dict)  # time_of_day, weather, location, source
    gt: dict = field(default_factory=dict)        # optional known labels (synthetic data only)
    sample_dir: Path | None = None

    @property
    def image_path(self) -> Path:
        assert self.sample_dir is not None, "sample has not been saved/loaded from disk"
        return self.sample_dir / "image.jpg"

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def image_bytes(self) -> bytes:
        return self.image_path.read_bytes()

    def lanes(self) -> list[np.ndarray]:
        """Centerlines of drivable lanes (bike lanes, LaneType 3, excluded)."""
        return [f.points for f in self.map_features if f.kind == "lane" and f.subtype != 3]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "segment_id": self.segment_id,
            "timestamp_us": int(self.timestamp_us),
            "camera": self.camera,
            "camera_model": self.camera_model.to_dict(),
            "ego_speed": None if self.ego_speed is None else round(float(self.ego_speed), 3),
            "context": self.context,
            "gt": self.gt,
            "boxes": [b.to_dict() for b in self.boxes],
            "map": [f.to_dict() for f in self.map_features],
        }


def make_sample_id(segment_id: str, timestamp_us: int, camera: str) -> str:
    return f"{segment_id}_{timestamp_us}_{camera}"


def save_sample(sample: Sample, image_jpeg: bytes, root: str | Path) -> Path:
    out = Path(root) / sample.sample_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "image.jpg").write_bytes(bytes(image_jpeg))
    (out / "sample.json").write_text(json.dumps(sample.to_dict(), separators=(",", ":")))
    sample.sample_dir = out
    return out


def load_sample(sample_dir: str | Path) -> Sample:
    sample_dir = Path(sample_dir)
    d = json.loads((sample_dir / "sample.json").read_text())
    return Sample(
        sample_id=d["sample_id"],
        segment_id=d["segment_id"],
        timestamp_us=int(d["timestamp_us"]),
        camera=d["camera"],
        camera_model=CameraModel.from_dict(d["camera_model"]),
        boxes=[Box3D.from_dict(b) for b in d["boxes"]],
        map_features=[MapFeature.from_dict(f) for f in d["map"]],
        ego_speed=d.get("ego_speed"),
        context=d.get("context", {}),
        gt=d.get("gt", {}),
        sample_dir=sample_dir,
    )


def iter_samples(root: str | Path) -> Iterator[Sample]:
    for path in sorted(Path(root).glob("*/sample.json")):
        yield load_sample(path.parent)


def _round(a, decimals: int = 3) -> list:
    return np.round(np.asarray(a, dtype=np.float64), decimals).tolist()

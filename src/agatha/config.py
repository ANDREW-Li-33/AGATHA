"""Stage 1 configuration, loaded from TOML."""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .columns import CAMERA_IDS


@dataclass(frozen=True)
class RenderProfile:
    name: str
    max_long_edge: int
    line_width: int
    font_scale: float
    jpeg_quality: int


@dataclass(frozen=True)
class MotionConfig:
    stationary_mps: float
    velocity_baseline_s: float
    accel_baseline_s: float
    max_baseline_s: float


@dataclass(frozen=True)
class CheckConfig:
    residual_median_px: float
    residual_p95_px: float
    slow_residual_median_px: float
    slow_residual_p95_px: float
    slow_speed_mps: float
    border_margin_px: float
    gap_ms: float
    speed_spike_mps: float


@dataclass(frozen=True)
class Stage1Config:
    dataset_version: str
    root: Path
    outputs: Path
    cameras: tuple[int, ...]
    window_s: float
    stride_s: float
    fps: float
    context_s: float
    max_objects: int
    profile: str
    motion: MotionConfig
    rear_axle_to_front_m: float
    profiles: dict[str, RenderProfile]
    checks: CheckConfig
    preprocessing_version: str


def _code_hash() -> str:
    """Hash of this package's source. A commit won't do: sync.sh copies the working tree, edits and all."""
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(source.read_bytes() for source in sources)).hexdigest()[:12]


def load_config(path: str | Path, root: str | Path | None = None, outputs: str | Path | None = None,
                cameras: list[str] | None = None) -> Stage1Config:
    """Load TOML, expanding $VARS. `root` and `outputs` override the file, for tests. `cameras` (names) is --cameras."""
    path = Path(path)
    raw = path.read_bytes()
    cfg = tomllib.loads(raw.decode())
    stage1 = cfg["stage1"]
    camera_ids = tuple(CAMERA_IDS[name] for name in cameras or stage1["cameras"])
    # The cameras count separately because the flag can override them, and they change the tables.
    config_hash = hashlib.sha256(raw + repr(camera_ids).encode()).hexdigest()[:12]
    outputs = os.path.expandvars(str(outputs if outputs is not None else cfg["dataset"]["outputs"]))
    if "$" in outputs:  # an unset variable stays as text, which would put Dataset files under the working directory
        raise ValueError("outputs names an unset environment variable")
    return Stage1Config(
        dataset_version=cfg["dataset"]["version"],
        root=Path(os.path.expandvars(str(root if root is not None else cfg["dataset"]["root"]))),
        outputs=Path(outputs),
        cameras=camera_ids,
        window_s=stage1["window_s"],
        stride_s=stage1["stride_s"],
        fps=stage1["fps"],
        context_s=stage1["context_s"],
        max_objects=stage1["max_objects"],
        profile=stage1["profile"],
        motion=MotionConfig(**cfg["motion"]),
        rear_axle_to_front_m=cfg["ego"]["rear_axle_to_front_m"],
        profiles={name: RenderProfile(name=name, **p) for name, p in cfg["render"].items()},
        checks=CheckConfig(**cfg["checks"]),
        preprocessing_version=f"{__version__}+{_code_hash()}.{config_hash}",
    )

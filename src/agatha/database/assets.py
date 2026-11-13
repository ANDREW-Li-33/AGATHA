"""Plain FRONT previews with original frame timing, and private Supabase Storage.

Only __clean.jpg files are read. Stage 1/2 outputs are never modified. Encodes
and their small timing manifests live in the caller's separate Stage 3 folder.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

from ..columns import TIME
from ..files import write_atomic
from .ingest import canonical, digest, identity, read_json

BUCKET = "agatha-incidents"


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class Clip:
    path: Path
    sha256: str
    metadata: dict


def export_clip(segment_dir: Path, incident: dict, output_dir: Path, context_s: float = 1.0) -> Clip:
    """Export available clean frames; never substitute annotated footage for a missing frame."""
    segment_dir, output_dir = Path(segment_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(segment_dir):
        raise ValueError("Stage 3 output must be outside the source segment directory")
    if not math.isfinite(context_s) or context_s < 0:
        raise ValueError("clip context must be finite and nonnegative")
    meta = read_json(segment_dir / "segment.json")
    if incident["preprocessing_version"] != meta["preprocessing_version"]:
        raise ValueError("clip and report preprocessing versions disagree")
    start, end = max(0.0, incident["start_s"] - context_s), min(meta["last_t_s"], incident["end_s"] + context_s)
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("invalid clip range")
    frames = pd.read_parquet(segment_dir / "frames.parquet", columns=[TIME, "camera", "t_s"])
    front = frames[frames.camera == 1].sort_values(TIME)
    if front.empty or front[TIME].duplicated().any():
        raise ValueError("missing or duplicate FRONT frame keys")
    available = [(int(row.frame_timestamp_micros), float(row.t_s),
                  segment_dir / "render" / "FRONT" / f"{row.frame_timestamp_micros}__clean.jpg")
                 for row in front.itertuples()]
    available = [item for item in available if item[2].is_file()]
    if not available:
        raise ValueError("no clean FRONT frames; run Stage 1/2 rendering first")
    # Include the closest frame on each side of the context interval when available.
    # This also gives a zero-duration report between frames something truthful to show.
    before = [item for item in available if item[1] <= start]
    after = [item for item in available if item[1] >= end]
    lower = before[-1][1] if before else start
    upper = after[0][1] if after else end
    selected = [item for item in available if lower <= item[1] <= upper]
    if not selected or selected[0][1] > max(0, incident["start_s"]) + 0.05 or selected[-1][1] < min(
            meta["last_t_s"], incident["end_s"]) - 0.05:
        raise ValueError("clean frames do not cover the incident")
    timestamps = [item[0] for item in selected]
    metadata = {"camera": "FRONT", "annotated": False, "context_s": context_s,
                "requested_start_s": start, "requested_end_s": end,
                "start_s": selected[0][1], "end_s": selected[-1][1],
                "duration_s": (timestamps[-1] - timestamps[0]) / 1e6 + 0.1,
                "frame_count": len(selected), "frame_timestamps_micros": timestamps,
                "max_gap_s": max((b - a for a, b in zip(timestamps, timestamps[1:])), default=0) / 1e6,
                "preprocessing_version": meta["preprocessing_version"],
                "encoder_version": "front-clean-v1"}
    key = digest([meta["context_name"], metadata, [file_hash(item[2]) for item in selected]])
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target, manifest = output_dir / f"{key}.mp4", output_dir / f"{key}.json"
    if target.exists() and manifest.exists():
        cached = read_json(manifest)
        if cached["metadata"] == metadata and cached["sha256"] == file_hash(target):
            return Clip(target, cached["sha256"], metadata)
    with tempfile.TemporaryDirectory(prefix="encode-", dir=output_dir) as temp:
        temp = Path(temp)
        lines = ["ffconcat version 1.0"]
        durations = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])] + [0.1]
        # Safe relative names avoid ffconcat quoting problems with apostrophes in source paths.
        for number, ((_, _, source), duration) in enumerate(zip(selected, durations)):
            name = f"frame-{number:06d}.jpg"
            (temp / name).symlink_to(source)
            lines += [f"file '{name}'", "option framerate 1000000", f"duration {duration:.6f}"]
        lines += [f"file 'frame-{len(selected) - 1:06d}.jpg'", "option framerate 1000000"]
        script = temp / "frames.ffconcat"
        script.write_text("\n".join(lines) + "\n")
        encoded = temp / "clip.mp4"
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(script),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-level:v", "5.0", "-fps_mode", "passthrough",
            "-movflags", "+faststart", str(encoded)], capture_output=True, timeout=300)
        if result.returncode or not encoded.is_file() or not encoded.stat().st_size:
            raise RuntimeError("plain clip encoding failed")
        encoded.chmod(0o600)
        encoded.replace(target)
    sha256 = file_hash(target)
    write_atomic(manifest, (canonical({"sha256": sha256, "metadata": metadata}) + "\n").encode())
    manifest.chmod(0o600)
    return Clip(target, sha256, metadata)


class PrivateStorage:
    def __init__(self, client):
        self._client = client

    @classmethod
    def from_env(cls):
        from supabase import create_client

        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise ValueError("Set SUPABASE_URL and SUPABASE_SECRET_KEY in the server's private environment file")
        return cls(create_client(url, key))

    def _bucket(self):
        bucket = self._client.storage.get_bucket(BUCKET)
        if bucket.public is not False:
            raise ValueError("refusing to use a public Storage bucket")
        return self._client.storage.from_(BUCKET)

    def upload_clip(self, incident_id: str, clip: Clip) -> dict:
        bucket = self._bucket()
        # Same bytes, same key, across models/incidents. No expiring URL is stored in PostgreSQL.
        path = f"front/{clip.sha256}.mp4"
        if clip.path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("clip exceeds the private bucket's 50 MiB limit")
        if not bucket.exists(path):
            from storage3.exceptions import StorageApiError
            try:
                with clip.path.open("rb") as source:
                    bucket.upload(path, source, {"content-type": "video/mp4", "upsert": "false"})
            except StorageApiError as exc:
                # A simultaneous importer may have uploaded these exact content-addressed bytes.
                if exc.code != "Duplicate" or not bucket.exists(path):
                    raise
        return {"id": identity("asset", [incident_id, BUCKET, path]), "incident_id": incident_id,
                "kind": "source_video", "bucket": BUCKET, "object_path": path, "sha256": clip.sha256,
                "content_type": "video/mp4", "size_bytes": clip.path.stat().st_size, "metadata": clip.metadata}

    def signed_url(self, asset: dict, expires_s: int = 300) -> str:
        if asset["bucket"] != BUCKET or not 1 <= expires_s <= 900:
            raise ValueError("invalid private asset or URL lifetime")
        return self._bucket().create_signed_url(asset["object_path"], expires_s)["signedURL"]

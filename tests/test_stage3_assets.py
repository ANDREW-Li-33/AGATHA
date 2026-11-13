import subprocess
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from agatha.database.assets import BUCKET, PrivateStorage, export_clip
from agatha.database.ingest import prepare_import
from stage3_helpers import scene_files


def test_plain_clip_actual_timing_cache_and_source_files_untouched(tmp_path):
    folder = scene_files(tmp_path / "source's folder")
    incident = prepare_import(folder).incidents[0]
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    clip = export_clip(folder, incident, tmp_path / "stage3", context_s=0.2)
    assert clip.metadata["annotated"] is False
    assert clip.metadata["start_s"] == 0.2 and clip.metadata["end_s"] == 1.0
    assert clip.metadata["max_gap_s"] == 0.3
    assert clip.metadata["duration_s"] == pytest.approx(0.9)
    reader = imageio_ffmpeg.read_frames(str(clip.path), pix_fmt="rgb24")
    video = next(reader)
    rgb = np.frombuffer(next(reader), np.uint8).reshape(48, 64, 3)
    # Clean image at 0.2 s is BGR(50,50,100); annotated alternative is its inverse.
    assert np.abs(rgb.mean(axis=(0, 1)) - [100, 50, 50]).max() < 5
    reader.close()
    assert video["codec"] == "h264"
    stamp = clip.path.stat().st_mtime_ns
    assert export_clip(folder, incident, tmp_path / "stage3", context_s=0.2) == clip
    assert clip.path.stat().st_mtime_ns == stamp
    assert before == {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert clip.path.stat().st_mode & 0o777 == 0o600
    # Verify presentation timestamps across the dropped samples, not merely nominal FPS.
    output = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(clip.path), "-vf", "showinfo", "-f", "null", "-"],
                            capture_output=True, check=True).stderr.decode()
    import re
    pts = [float(s) for s in re.findall(r"pts_time:([0-9.]+)", output)]
    assert pts[:5] == pytest.approx([0, 0.2, 0.3, 0.6, 0.8], abs=0.00001)


def test_missing_clean_frames_never_uses_annotated_clip(tmp_path):
    folder = scene_files(tmp_path)
    incident = prepare_import(folder).incidents[0]
    for p in (folder / "render/FRONT").glob("*__clean.jpg"):
        p.unlink()
    with pytest.raises(ValueError, match="no clean"):
        export_clip(folder, incident, tmp_path / "stage3")
    assert not (tmp_path / "stage3").exists()


@pytest.mark.parametrize("context", [-1, float("nan"), float("inf")])
def test_clip_rejects_bad_context(tmp_path, context):
    folder = scene_files(tmp_path)
    with pytest.raises(ValueError):
        export_clip(folder, prepare_import(folder).incidents[0], tmp_path / "stage3", context)


def test_clip_output_cannot_modify_stage1_directory(tmp_path):
    folder = scene_files(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        export_clip(folder, prepare_import(folder).incidents[0], folder / "clips")


class Bucket:
    def __init__(self):
        self.files = {}
        self.uploads = 0
        self.signed = []
    def exists(self, path):
        return path in self.files
    def upload(self, path, source, options):
        assert options == {"content-type": "video/mp4", "upsert": "false"}
        self.files[path] = source.read()
        self.uploads += 1
    def create_signed_url(self, path, seconds):
        self.signed.append((path, seconds))
        return {"signedURL": "https://synthetic.invalid/private?token=temporary"}


def test_private_content_addressed_upload_and_short_signed_urls(tmp_path):
    folder = scene_files(tmp_path)
    incident = prepare_import(folder).incidents[0]
    clip = export_clip(folder, incident, tmp_path / "stage3")
    bucket, info = Bucket(), SimpleNamespace(public=False)
    client = SimpleNamespace(storage=SimpleNamespace(get_bucket=lambda _: info, from_=lambda _: bucket))
    storage = PrivateStorage(client)
    first = storage.upload_clip(incident["id"], clip)
    assert first == storage.upload_clip(incident["id"], clip)
    assert bucket.uploads == 1 and first["bucket"] == BUCKET
    assert "token=" in storage.signed_url(first)
    assert bucket.signed[-1][1] == 300
    assert "token=" not in str(first)
    info.public = True
    with pytest.raises(ValueError, match="public"):
        storage.signed_url(first)
    with pytest.raises(ValueError, match="public"):
        storage.upload_clip(incident["id"], clip)
    with pytest.raises(ValueError):
        storage.signed_url(first, 3600)

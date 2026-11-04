"""One way to write a file that something later will trust."""

from __future__ import annotations

from pathlib import Path


def write_atomic(path, data: bytes) -> None:
    """Replace in one step, creating the directory, so a killed job can't leave half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(data)
    partial.replace(path)

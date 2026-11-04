"""The data boundary: the report is the only file that leaves PACE, so this fails closed.

Only numbers and a fixed vocabulary of strings get in, and every name has a fixed shape, so
an object ID or context name can't slip in by accident. Errors raised here never echo what
they rejected, because they can reach a Slurm log.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .columns import ROLLING_SHUTTER
from .files import write_atomic

PUBLIC_SEGMENT = "10023947602400723454_1120_000_1140_000"  # Waymo's tutorial segment
VOCABULARY = frozenset({*ROLLING_SHUTTER.values(), "pass", "fail", "no_data"})

# Aggregates stay well inside both limits, and a leak breaks one: a raw timestamp in ms or
# micros is over 1e12, and a list is longest at one entry per window.
_MAX_NUMBER = 1e12
_MAX_LIST = 32

# Always fullmatch: re.match with $ would also accept a trailing newline.
_NAME = re.compile(r"[a-z][a-z0-9_]*")
_VERSION = re.compile(r"[0-9A-Za-z.+-]+")
# The whole name, not its start: synthetic_x/../<name> would pass a prefix check and put <name> in the report.
_SYNTHETIC = re.compile(r"synthetic_[A-Za-z0-9_]+")


def context_label(context_name: str) -> str:
    """The name if it's public or synthetic, otherwise a short hash of it."""
    if context_name == PUBLIC_SEGMENT or _SYNTHETIC.fullmatch(context_name):
        return context_name
    return "sha256:" + hashlib.sha256(context_name.encode()).hexdigest()[:12]


def _name(name) -> str:
    if isinstance(name, str) and _NAME.fullmatch(name):
        return name
    # from None: fail() runs in except blocks, where the real exception and its message
    # would otherwise print as this one's context.
    raise ValueError("report names must match [a-z][a-z0-9_]*") from None


def _clean(value):
    """A JSON-safe copy of `value`, with non-finite floats as None, or a ValueError."""
    if isinstance(value, np.generic):
        value = value.item()  # aggregates computed by NumPy are numbers like any other
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        if abs(value) >= _MAX_NUMBER:
            raise ValueError("report numbers must be aggregates, and this one is the size of a raw timestamp")
        return value
    if isinstance(value, str) and value in VOCABULARY:
        return value
    if isinstance(value, list):
        if len(value) > _MAX_LIST:
            raise ValueError("report lists must be short, and this one is the length of a column")
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {_name(key): _clean(item) for key, item in value.items()}
    raise ValueError(f"report can't hold this {type(value).__name__}")


def _location(exc: BaseException) -> str | None:
    """file:line of the innermost frame in this package. Just the file: full paths name the user."""
    location = None
    for frame, line in traceback.walk_tb(exc.__traceback__):
        if frame.f_globals.get("__package__") == __package__:
            location = f"{Path(frame.f_code.co_filename).name}:{line}"
    return location


class Report:
    def __init__(self, context_name: str, preprocessing_version: str):
        # The one free-form string in the file. Swapped arguments would write the raw context
        # name, and nothing would notice on synthetic data, whose names are public.
        if not _VERSION.fullmatch(preprocessing_version):
            raise ValueError("preprocessing_version has an unexpected shape")
        self.context_label = context_label(context_name)
        self.preprocessing_version = preprocessing_version
        self.sections: dict[str, dict] = {}
        self.failures: list[dict] = []

    def add(self, section: str, **metrics) -> None:
        """Merge aggregates into a section. Cleaned first, so a rejected call adds nothing."""
        clean = _clean(metrics)
        self.sections.setdefault(_name(section), {}).update(clean)

    def fail(self, step: str, exc: BaseException) -> None:
        """Record where a step failed. Never str(exc): messages can echo Dataset values."""
        self.failures.append({"step": _name(step), "exception": type(exc).__name__,
                              "location": _location(exc)})

    def write(self, path) -> Path:
        """Write the report, creating its directory: a run can fail before anything else does."""
        document = {"context": self.context_label, "preprocessing_version": self.preprocessing_version,
                    "sections": self.sections, "failures": self.failures}
        write_atomic(path, (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        return Path(path)


@contextmanager
def guarded(report: Report, step: str) -> Iterator[None]:
    """Run a step. On any exception, record it and exit 1, with no message or traceback.

    BaseException, because a SystemExit carrying a string prints that string on the way out.
    `from None` keeps the original out of anything that formats this exit. Writing the file
    is the caller's job, in a finally.
    """
    _name(step)  # up front: a ValueError from inside the handler would hide the real failure
    try:
        yield
    except BaseException as exc:
        report.fail(step, exc)
        raise SystemExit(1) from None

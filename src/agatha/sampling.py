"""Gap reports, frame selection, and window arithmetic over real timestamps.

A segment is nominally 20 s at 10 Hz, but frames can be missing and spacing
varies, so nothing here assumes a frame count or uses row positions. Timestamps
stay int64 throughout. Seconds only describe ranges and grid times.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Half a 10 Hz frame. Spacing jitters and drifts, so the last frame can land just short
# of its grid time, and a strict comparison with end_s = last_t_s would then drop it.
_CLOSED_END_SLACK_US = 50_000


class GapReport(NamedTuple):
    n_frames: int
    duration_s: float
    min_dt_ms: float
    median_dt_ms: float
    max_dt_ms: float
    n_gaps: int


class Selection(NamedTuple):
    timestamps: np.ndarray  # int64, sorted, unique
    achieved_fps: float     # intervals / span of the selected frames, 0 if fewer than 2
    max_gap_ms: float       # largest spacing within the selection, 0 if fewer than 2


class Window(NamedTuple):
    start_s: float
    end_s: float
    closed_end: bool        # True only for a segment's last window


def gap_report(timestamps, gap_ms) -> GapReport:
    """Spacing summary, in place of any fixed frame count. Spacing is NaN below 2 frames."""
    ts = np.unique(np.asarray(timestamps, dtype=np.int64))
    if len(ts) < 2:
        return GapReport(len(ts), 0.0, np.nan, np.nan, np.nan, 0)
    dt_ms = np.diff(ts) / 1e3
    return GapReport(len(ts), float((ts[-1] - ts[0]) / 1e6), float(dt_ms.min()), float(np.median(dt_ms)),
                     float(dt_ms.max()), int(np.count_nonzero(dt_ms > gap_ms)))


def nearest_frame(timestamps, t0_micros, t_s) -> int:
    """The exact timestamp closest to t_s, for force_include: t0 plus t_s in microseconds names no frame."""
    ts = np.asarray(timestamps, dtype=np.int64)
    return int(ts[np.abs(ts - (t0_micros + round(t_s * 1e6))).argmin()])


def select_frames(timestamps, t0_micros, fps, start_s, end_s, closed_end=False, force_include=()) -> Selection:
    """Take the nearest frame to each grid time, never every Nth row.

    The grid is anchored at t0, not at start_s, so overlapping windows share frames.
    `timestamps` can repeat, since frames.parquet has one row per camera. A grid time
    with no frame within half a period selects nothing, so a hole stays a hole.

    start_s and end_s bound grid times, not frames: jitter can put a frame just outside
    them, and force_include can name any frame there is. It must be a real one, though.
    """
    ts = np.unique(np.asarray(timestamps, dtype=np.int64))
    # The cast matters: union1d with an empty tuple would turn the timestamps into float64.
    forced = np.asarray(force_include, dtype=np.int64)
    if np.setdiff1d(forced, ts).size:  # a value that's no frame would count in the spacing and select nothing
        raise ValueError("force_include must hold exact frame timestamps")
    period_us = 1e6 / fps
    # Whole microseconds, because in floats 197 * 0.1 > 19.7 and a frame falls off the end.
    start_us, end_us = round(start_s * 1e6), round(end_s * 1e6)
    grid = np.round(np.arange(int(end_s * fps) + 2) * period_us).astype(np.int64)
    before_end = grid <= end_us + _CLOSED_END_SLACK_US if closed_end else grid < end_us
    targets = t0_micros + grid[(grid >= start_us) & before_end]

    nearest = ts[np.abs(ts[:, None] - targets).argmin(axis=0)]  # on a tie, argmin takes the earlier frame
    chosen = nearest[np.abs(nearest - targets) <= period_us / 2]
    selected = np.union1d(chosen, forced)
    if len(selected) < 2:
        return Selection(selected, 0.0, 0.0)
    span_s = (selected[-1] - selected[0]) / 1e6
    return Selection(selected, float((len(selected) - 1) / span_s), float(np.diff(selected).max() / 1e3))


def compute_windows(last_t_s, window_s, stride_s, min_tail_s=1.0) -> list[Window]:
    """Windows from 0 to the real last timestamp, which is rarely a round 20 s."""
    windows, start = [], 0.0
    while start + window_s < last_t_s:
        windows.append(Window(start, start + window_s, False))
        start = len(windows) * stride_s
    # A tail is the time past the previous window, not the last window's own length:
    # with overlap that length always exceeds window_s - stride_s, so nothing would merge.
    if windows and last_t_s - windows[-1].end_s < min_tail_s:
        start = windows.pop().start_s
    windows.append(Window(start, last_t_s, True))
    return windows

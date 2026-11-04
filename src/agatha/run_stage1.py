"""The Stage 1 CLI: one segment from Parquet components to tables, windows, images, and a report.

Real data passes through here, so this prints only counts and the report's path, which holds
the context's label rather than its name. Every step runs in `guarded`, which keeps exception
messages out of Slurm logs. The report is written however the run ends, short of SIGKILL, and
the old one goes first, so no report means no finished run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .checks import (
    box_residuals, check_ego_velocity, check_gaps, check_joins, check_residuals, check_tracks, check_velocity,
    null_summary, output_hashes, pick_review_frames, timing_summary,
)
from .columns import CAMERA_IDS, CAMERA_NAMES, REQUIRED_COLUMNS
from .config import RenderProfile, Stage1Config, load_config
from .contracts import SceneWindow
from .loader import (
    SegmentRef, check_segment, load_cameras, load_image_metadata, load_lidar_boxes, load_poses,
    load_projected_boxes, load_synced_boxes, read_jpegs,
)
from .render import ContactItem, boxes_for_image, ensure_rendered, render_annotated, write_clip, write_contact_sheet
from .report import Report, guarded
from .scene_window import default_windows, write_scene_window
from .store import (
    SegmentOutputs, build_frames_table, build_segment_meta, camera_models, load_segment, segment_directory,
    write_segment,
)
from .synthetic import write_synthetic_segment
from .tracks import build_ego_table, build_tracks_table


class _QuietParser(argparse.ArgumentParser):
    def error(self, message):
        """argparse would quote the bad argument, and that can be a context name."""
        self.exit(2, f"{self.prog}: bad arguments, see --help\n")


def _check_schema(root: Path, ref: SegmentRef, report: Report) -> None:
    """Footers only, so a bad file stops the run before anything loads.

    The report takes no names, so a missing column is its position in REQUIRED_COLUMNS[tag].
    """
    checks = check_segment(root, ref)
    passed = all(check.ok for check in checks.values())
    report.add("schema", status="pass" if passed else "fail", components={tag: {
        "missing": len(check.missing),
        "missing_at": [position for position, name in enumerate(REQUIRED_COLUMNS[tag]) if name in check.missing],
        "missing_optional": len(check.missing_optional),
        "extra": check.extra,
    } for tag, check in checks.items()})
    if not passed:
        raise ValueError("a component failed its schema check")


def _clear_outputs(directory: Path, preprocessing_version: str) -> None:
    """Remove what the new tables would leave stale, before they replace the old ones.

    Windows, clips, and the contact sheet always go. Images stay if the same code and config
    made them, because that cache is what makes a rerun cheap. With no segment.json to say
    which made them, they go too.
    """
    meta = directory / "segment.json"
    stale = ["windows", "clips", "qa"]
    if not meta.exists() or json.loads(meta.read_text())["preprocessing_version"] != preprocessing_version:
        stale.append("render")
    for name in stale:
        shutil.rmtree(directory / name, ignore_errors=True)


def _write_clips(window: SceneWindow, stem: str, segment: SegmentOutputs) -> int:
    """One clip per camera, from its frames' real times: a frame for grid time 6.0 can sit at 5.9998 s."""
    cameras = sorted({frame.camera for frame in window.frames})
    for camera in cameras:
        times = [frame.t_s for frame in window.frames if frame.camera == camera]
        write_clip(segment, CAMERA_IDS[camera], min(times), max(times),
                   segment.directory / "clips" / f"{stem}__{camera}.mp4", window.profile)
    return len(cameras)


def _review_items(picks: list[tuple[int, int, str]], segment: SegmentOutputs, root: Path, ref: SegmentRef,
                  profile: RenderProfile) -> list[ContactItem]:
    """The review frames, annotated. They can be in no window, so the render cache can't serve them."""
    models = camera_models(segment.meta)
    jpegs = {camera: read_jpegs(root, ref, camera, [ts for ts, picked, _ in picks if picked == camera])
             for camera in {camera for _, camera, _ in picks}}
    items = []
    for timestamp, camera, reason in picks:
        boxes = boxes_for_image(segment, timestamp, camera, models[camera])
        t_s = (timestamp - segment.meta["t0_micros"]) / 1e6
        items.append(ContactItem(render_annotated(jpegs[camera][timestamp], boxes, models[camera], profile),
                                 f"{CAMERA_NAMES[camera]} at {t_s:.2f} s: {reason}"))
    return items


def _run_steps(config: Stage1Config, ref: SegmentRef, directory: Path, report: Report) -> None:
    root, camera_ids = config.root, config.cameras

    with guarded(report, "schema"):
        _check_schema(root, ref, report)

    with guarded(report, "load"):
        poses = load_poses(root, ref)
        cameras = load_cameras(root, ref, camera_ids)
        image_meta = load_image_metadata(root, ref, camera_ids)
        boxes = load_lidar_boxes(root, ref)
        synced = load_synced_boxes(root, ref)
        projected = load_projected_boxes(root, ref, camera_ids)
        report.add("load", poses=len(poses.timestamps), cameras=len(cameras), images=len(image_meta),
                   lidar_boxes=len(boxes), synced_boxes=len(synced), projected_boxes=len(projected))

    # Before the tables, not with the other checks: an orphan stops them, and this says how many.
    with guarded(report, "joins"):
        report.add("joins", **check_joins(poses, boxes, synced, projected, image_meta, camera_ids))

    with guarded(report, "tables"):
        ego, path = build_ego_table(poses, config.motion)
        tracks = build_tracks_table(boxes, synced, projected, poses, ego, path, config.motion,
                                    config.rear_axle_to_front_m)
        frames = build_frames_table(image_meta, poses, config.checks.gap_ms)
        _clear_outputs(directory, config.preprocessing_version)
        write_segment(directory, build_segment_meta(ref, poses, cameras, config), frames, ego, tracks)
        segment = load_segment(directory)  # what's on disk is what every later stage gets
        report.add("tables", frames=len(segment.frames), ego=len(segment.ego), tracks=len(segment.tracks))

    with guarded(report, "windows"):
        windows = default_windows(segment, config)
        stems = [write_scene_window(window, directory).stem for window in windows]
        report.add("windows", windows=len(windows), images=[len(window.frames) for window in windows],
                   objects=[len(window.objects) for window in windows],
                   omitted_objects=[len(window.omitted_objects) for window in windows],
                   achieved_fps=[window.achieved_fps for window in windows],
                   max_gap_ms=[window.max_gap_ms for window in windows])

    with guarded(report, "render"):
        drawn = Counter()
        for window in windows:
            drawn.update(ensure_rendered(window, segment, root, ref, config.profiles))
        clips = sum(_write_clips(window, stem, segment) for window, stem in zip(windows, stems))
        report.add("render", clips=clips, **drawn)

    with guarded(report, "checks"):
        report.add("gaps", **check_gaps(poses, image_meta, config))
        residuals = box_residuals(segment, projected, cameras, config)
        report.add("residuals", **check_residuals(residuals, config))
        report.add("tracks", **check_tracks(segment, config))
        report.add("velocity", **check_velocity(segment))
        report.add("ego_velocity", **check_ego_velocity(segment, image_meta))
        report.add("timing", **timing_summary(segment.frames, cameras))
        report.add("nulls", **null_summary(boxes))

    with guarded(report, "contact_sheet"):
        picks = pick_review_frames(residuals, segment)
        items = _review_items(picks, segment, root, ref, config.profiles[config.profile])
        write_contact_sheet(directory / "qa" / "contact_sheet.html", items)
        report.add("contact_sheet", frames=len(picks), **Counter(reason for _, _, reason in picks))


def _statuses(report: Report) -> Counter:
    return Counter(section["status"] for section in report.sections.values() if "status" in section)


def _report_path(config: Stage1Config, report: Report) -> Path:
    """Named by label, so a session can fetch a report without listing the context names beside it."""
    return config.outputs / "reports" / f"{report.context_label}.json"


def _write_report(report: Report, path: Path) -> None:
    """Write the report and print its counts and path."""
    with guarded(report, "report"):
        report.write(path)
        statuses = _statuses(report)
        print(f"checks: {statuses['pass']} pass, {statuses['fail']} fail, {statuses['no_data']} no_data; "
              f"failed steps: {len(report.failures)}")
        print(f"report: {path}")


def process_segment(config: Stage1Config, ref: SegmentRef) -> tuple[Path, Report]:
    """Run every step on one segment. A failed step exits 1."""
    directory = segment_directory(config, ref)
    report = Report(ref.context_name, config.preprocessing_version)
    path = _report_path(config, report)
    try:
        with guarded(report, "start"):
            path.unlink(missing_ok=True)
        _run_steps(config, ref, directory, report)
    finally:
        _write_report(report, path)
    return directory, report


def _repro_check(config: Stage1Config, ref: SegmentRef) -> tuple[Path, Report]:
    """Check 6: two runs, and only whether their hashes match.

    Both start empty, because ensure_rendered trusts what's on disk: a run over an older one
    would reuse its images and prove nothing about them.
    """
    base = config.outputs / "repro_check"
    shutil.rmtree(base, ignore_errors=True)
    first_config = replace(config, outputs=base / "first")
    first, report = process_segment(first_config, ref)
    second, _ = process_segment(replace(config, outputs=base / "second"), ref)
    try:
        with guarded(report, "repro"):
            matched = output_hashes(first) == output_hashes(second)
            report.add("repro", status="pass" if matched else "fail", hashes_match=matched)
            print(f"repro: hashes {'match' if matched else 'differ'}")
    finally:
        _write_report(report, _report_path(first_config, report))
    return first, report


def main(argv=None) -> int:
    os.umask(0o077)  # before anything is written, so that privacy doesn't rest on the caller sourcing env.sh
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # Slurm's time limit: `guarded` and the finally still run
    # For what no step guards, like a bad config: a traceback prints paths, and those can hold the context.
    sys.excepthook = lambda *_: print("agatha-stage1: failed outside any step", file=sys.stderr)
    parser = _QuietParser(prog="agatha-stage1", description="Run Stage 1 on one segment.")
    parser.add_argument("--config", default="configs/stage1.toml")
    parser.add_argument("--split", default="training")
    parser.add_argument("--context", default="synthetic_cut_in")
    parser.add_argument("--cameras", nargs="+", choices=list(CAMERA_IDS), help="override the config's cameras")
    parser.add_argument("--synthetic", action="store_true",
                        help="first write a synthetic segment under the outputs root")
    parser.add_argument("--repro-check", action="store_true",
                        help="process twice under <outputs>/repro_check, leaving other outputs alone")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.context):  # a path, like synthetic_x/../<name>, could pass for synthetic
        parser.error("context")

    config = load_config(args.config, cameras=args.cameras)
    ref = SegmentRef(args.split, args.context)
    if args.synthetic:
        config = replace(config, root=config.outputs / "synthetic")
        write_synthetic_segment(config.root, ref.split, ref.context_name)
    _, report = (_repro_check if args.repro_check else process_segment)(config, ref)
    return 1 if _statuses(report)["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())

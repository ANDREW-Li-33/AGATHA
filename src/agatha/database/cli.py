"""Stage 3 commands: migrate, validate, import one scene or every model in a scene list."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .assets import PrivateStorage, export_clip
from .ingest import prepare_import, result_directories
from .repository import Repository


def import_scene(segment_dir: Path, results: list[Path], *, repository=None, storage=None, output_dir=None,
                 context_s=1.0, human_path=None, scene=None) -> dict:
    if not results:
        raise ValueError("scene has no Stage 2 results")
    # All model variants validate before the first upload or insert.
    bundles = [prepare_import(segment_dir, result, human_path=human_path, scene=scene) for result in results]
    incidents, assets = set(), 0
    for bundle in bundles:
        if repository:
            repository.ingest(bundle)
        for incident in bundle.incidents:
            if incident["id"] in incidents:
                continue
            incidents.add(incident["id"])
            if output_dir:
                clip = export_clip(segment_dir, incident, output_dir, context_s)
                if storage:
                    repository.add_asset(storage.upload_clip(incident["id"], clip))
                assets += 1
    return {"runs": len({b.run["id"] for b in bundles}), "incidents": len(incidents), "clips": assets}


def main(argv=None) -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog="agatha-stage3", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--sql-dir", type=Path, default=Path("infra/sql"))
    migrate.add_argument("--without-storage", action="store_true", help="plain local PostgreSQL tests only")
    for command in ("validate", "ingest"):
        p = sub.add_parser(command)
        selection = p.add_mutually_exclusive_group(required=True)
        selection.add_argument("--segment", type=Path)
        selection.add_argument("--list", type=Path, help="one context per line, in review-page order")
        p.add_argument("--outputs", type=Path, help="Stage 1 outputs root, required with --list")
        p.add_argument("--stage2", type=Path, help="one model's results; default imports all saved models")
        p.add_argument("--human-labels", type=Path, help="one JSON file, or a labels directory for --list")
        p.add_argument("--scene", help="expected human-label scene name, required with one label file")
        p.add_argument("--clips", type=Path, help="separate Stage 3 output directory; validate exports locally")
        p.add_argument("--context-s", type=float, default=1.0)
        if command == "ingest":
            p.add_argument("--metadata-only", action="store_true", help="import reports without uploading clips")
    args = parser.parse_args(argv)
    if args.command != "migrate":
        if args.list and (not args.outputs or args.stage2 or args.scene):
            parser.error("--list needs --outputs and cannot use --stage2 or --scene")
        if args.segment and args.human_labels and not args.scene:
            parser.error("a human label file needs --scene")
        if args.command == "ingest" and not args.metadata_only and not args.clips:
            parser.error("ingest needs --clips or explicit --metadata-only")
    try:
        if args.command == "migrate":
            applied = Repository.from_env().migrate(args.sql_dir, storage=not args.without_storage)
            print(f"Stage 3: {len(applied)} migrations applied")
            return 0
        repository = Repository.from_env() if args.command == "ingest" else None
        storage = PrivateStorage.from_env() if repository and not args.metadata_only else None
        if args.list:
            contexts = args.list.read_text().split()
            if not contexts or any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in contexts):
                raise ValueError("scene list is empty or has invalid context names")
            prefix = "cutin" if args.list.name == "cutin_top.txt" else args.list.stem
            scenes = [(args.outputs / context, f"{prefix}_{rank}") for rank, context in enumerate(contexts, 1)]
        else:
            scenes = [(args.segment, args.scene)]
        failed = 0
        for directory, scene in scenes:
            try:
                labels = (args.human_labels / f"{scene}.json" if args.list and args.human_labels else args.human_labels)
                # If a label directory was explicitly requested, missing labels are errors, never silently skipped.
                counts = import_scene(directory, [args.stage2] if args.stage2 else result_directories(directory),
                                      repository=repository, storage=storage, output_dir=args.clips,
                                      context_s=args.context_s, human_path=labels, scene=scene)
                print(f"{scene or 'scene'}: {counts['runs']} runs, {counts['incidents']} incidents, {counts['clips']} clips")
            except Exception as exc:
                failed += 1
                # Driver errors can echo a connection URL or token. Never print their messages.
                print(f"{scene or 'scene'}: failed ({type(exc).__name__}); check inputs and private configuration",
                      file=sys.stderr)
        print(f"Stage 3: {len(scenes) - failed} scenes ready, {failed} failed")
        return int(bool(failed))
    except Exception as exc:
        print(f"Stage 3: failed ({type(exc).__name__}); check inputs and private configuration", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Pick scenes at random from the labeled segments, for Stage 2 and Andrew to label: $WOD_HOME/selected/random15.txt.

Run on PACE from ~/agatha with AGATHA's environment, after download_labels.sbatch: source infra/pace/env.sh, then
uv run --frozen python infra/pace/pick_random.py. The labeled segments are the lidar_box files of training and
validation, less every segment on the exclude lists, by default the scan's two. The seed is fixed, and an existing
list is kept unless --force, so the set stays the one Andrew labels. It writes one name a line, in pick order,
and prints only counts.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

from agatha.files import write_atomic

# The scan's own lists, and names whose prefix a folder in $WOD_HOME/review already has: cutin is
# cutin_top.txt's, review/cutins holds the scan's pages, which agatha-scan review clears, and review/stage2
# holds Stage 2's links.
_TAKEN = {"cutin_top.txt", "fast_left.txt", "cutin.txt", "cutins.txt", "stage2.txt"}


def labeled(root: Path) -> list[str]:
    """Every segment with labels: the stems of lidar_box in both labeled splits, sorted so only the seed decides."""
    return sorted({path.stem for split in ("training", "validation")
                   for path in (root / split / "lidar_box").glob("*.parquet")})


def main(argv=None) -> int:
    os.umask(0o077)  # before anything is written: the list names contexts
    parser = argparse.ArgumentParser(prog="pick_random.py", description="Pick labeled segments at random.")
    parser.add_argument("--n", type=int, default=15, help="how many to pick")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--exclude", nargs="*", default=["cutin_top.txt", "fast_left.txt"],
                        help="lists in $WOD_HOME/selected whose segments are never picked")
    parser.add_argument("--out", default="random15.txt", help="the list to write, in $WOD_HOME/selected")
    parser.add_argument("--force", action="store_true", help="replace the list if it's there already")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9_]+\.txt", args.out) or args.out in _TAKEN:  # the PACE scripts' rule, less those
        parser.error("--out takes a new list's name, such as random15.txt")
    if args.n < 1:
        parser.error("--n takes 1 or more")
    selected = Path(os.environ["WOD_HOME"]) / "selected"
    missing = [listing for listing in args.exclude if not (selected / listing).is_file()]
    if missing:  # a list left out would let its segments be picked again
        parser.error(f"no list {', '.join(missing)} in $WOD_HOME/selected")
    out = selected / args.out
    if out.exists() and not args.force:
        print(f"pick_random.py: {args.out} is there already, so it stays: --force replaces it", file=sys.stderr)
        return 1
    excluded = {name for listing in args.exclude for name in (selected / listing).read_text().split()}
    segments = labeled(Path(os.environ["WOD_ROOT"]))
    pool = [name for name in segments if name not in excluded]
    if len(pool) < args.n:
        print(f"pick_random.py: only {len(pool)} labeled segments left to pick from", file=sys.stderr)
        return 1
    picked = random.Random(args.seed).sample(pool, args.n)
    write_atomic(out, "".join(f"{name}\n" for name in picked).encode())
    print(f"{args.out}: {len(picked)} picked from {len(pool)} labeled segments, "
          f"{len(segments) - len(pool)} left out as listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

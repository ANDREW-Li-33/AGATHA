#!/bin/bash
#SBATCH -J review-cutins
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 4:00:00
# usage: review_cutins.sbatch <list>.txt
# Run order: download_labels.sbatch, then scan.sbatch, then download_listed.sbatch for each list
# (cutin_top.txt, fast_left.txt), then review_cutins.sbatch for each list.
# A list is $WOD_HOME/selected/<list>.txt, named in [a-z0-9_]: the scan's two, or one pick_random.py
# wrote, such as random15.txt, whose pages agatha-pages writes once Stage 2 has run.
# Stage 1 with three cameras on each listed segment that has camera data. For cutin_top.txt, a review
# page per candidate follows, in $WOD_HOME/review/cutins/cutin_<rank>.html. For fast_left.txt, Stage 1's
# own checks are the point: they judge FRONT_LEFT at speed.
# Segments print by rank, as <prefix>_<rank>, never by name: cutin for cutin_top.txt, as its pages are
# named, and otherwise the list's stem, so fast_left_<rank> and random15_<rank>. agatha-stage1's own
# output goes nowhere: each report path it prints holds a label, and a list of labels is a context list.
# Up to 20 segments, twice --top, or random15.txt's 15. Single-segment Stage 1 jobs took under 2
# minutes on ICE, so 4 hours leaves wide room.
set -euo pipefail
source infra/pace/env.sh
if ! [[ ${1:-} =~ ^[a-z0-9_]+\.txt$ ]]; then  # a name, never a path
  echo "usage: review_cutins.sbatch <list>.txt, a list in \$WOD_HOME/selected" >&2
  exit 2
fi
case $1 in
  cutin_top.txt) prefix=cutin ;;
  *) prefix=${1%.txt} ;;
esac
rank=0
# The list comes in on descriptor 3: ffmpeg reads stdin, so on stdin it would eat the next lines.
while IFS= read -r context <&3; do
  rank=$((rank + 1))
  for split in training validation; do
    if [ -f "$WOD_ROOT/$split/camera_image/$context.parquet" ]; then
      status=0
      uv run --frozen agatha-stage1 --config configs/stage1.toml --split "$split" --context "$context" \
        --cameras FRONT FRONT_LEFT FRONT_RIGHT > /dev/null 2>&1 || status=$?
      echo "${prefix}_$rank: agatha-stage1 exit $status"
    fi
  done
done 3< "$WOD_HOME/selected/$1"
if [ "$1" = cutin_top.txt ]; then
  uv run --frozen agatha-scan review --config configs/stage1.toml --scan-config configs/scan.toml
fi

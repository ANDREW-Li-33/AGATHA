#!/bin/bash
#SBATCH -J stage2-cutin
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 2:00:00
# usage: stage2_cutin.sbatch <rank> [claude|dry-run]
# Stage 2 on the segment at <rank> in cutin_top.txt (1 is the demo, page 1), after review_cutins.sbatch
# has run Stage 1 on it. dry-run, the default, sends nothing. claude needs real_data_cleared in
# configs/stage2.toml and the key in ~/.config/agatha/anthropic.env, outside ~/agatha so sync.sh never
# touches it. Inside a running job: srun --jobid=<job> --overlap bash infra/pace/stage2_cutin.sbatch 1
# VLM_CONFIG=<path> runs with another stage2.toml, as for trying another model. The default is
# configs/stage2.toml.
# Stage 2's outputs, under the segment's stage2/, describe a real scene, so they stay on PACE: Andrew
# reviews them in OnDemand, through the link $WOD_HOME/review/stage2/cutin_<rank>. Only counts print,
# with cutin_<rank> in place of the label, a hash of the name, and of the name itself.
set -euo pipefail
source infra/pace/env.sh
rank=${1:-}
model=${2:-dry-run}
if ! [[ $rank =~ ^[1-9][0-9]*$ && $model =~ ^(claude|dry-run)$ ]]; then
  echo "usage: stage2_cutin.sbatch <rank> [claude|dry-run]" >&2
  exit 2
fi
context=$(sed -n "${rank}p" "$WOD_HOME/selected/cutin_top.txt")
if ! [[ $context =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "cutin_$rank: no such rank" >&2
  exit 2
fi
split=
for candidate in training validation; do
  if [ -f "$WOD_ROOT/$candidate/camera_image/$context.parquet" ]; then
    split=$candidate
  fi
done
if [ -z "$split" ]; then
  echo "cutin_$rank: no camera data" >&2
  exit 1
fi
if [ "$model" = claude ]; then
  source "$HOME/.config/agatha/anthropic.env"  # exports the key, for agatha-vlm only. Never echo it.
fi
status=0
uv run --frozen --extra claude agatha-vlm --config configs/stage1.toml --vlm-config "${VLM_CONFIG:-configs/stage2.toml}" \
  --split "$split" --context "$context" --model "$model" 2>&1 \
  | sed -E -e "s/sha256:[0-9a-f]{12}/cutin_$rank/g" -e "s/$context/cutin_$rank/g" || status=$?
echo "cutin_$rank: agatha-vlm exit $status"
if [ "$status" = 0 ]; then  # the results by page number, as the review pages are
  mkdir -p "$WOD_HOME/review/stage2"
  ln -sfn "$WOD_OUTPUTS/$context/stage2" "$WOD_HOME/review/stage2/cutin_$rank"
fi

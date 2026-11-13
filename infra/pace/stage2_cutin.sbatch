#!/bin/bash
#SBATCH -J stage2-cutin
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 2:00:00
# usage: [LIST=<list>.txt] stage2_cutin.sbatch <rank> [local|claude|dry-run]
# Stage 2 on the segment at <rank> in a scene list, $WOD_HOME/selected/$LIST, named in [a-z0-9_]. LIST is
# cutin_top.txt by default, where rank 1 is the demo, page 1; random15.txt is another. Run it once
# review_cutins.sbatch has run Stage 1 on that list. The scene is <prefix>_<rank>: cutin_<rank> for
# cutin_top.txt, as its pages are named, and otherwise the list's stem, as in random15_<rank>.
# dry-run, the default, sends nothing. claude needs real_data_cleared in configs/stage2.toml and the key
# in ~/.config/agatha/anthropic.env, outside ~/agatha so sync.sh never touches it. local asks Gemma 4 on
# this node, so it runs inside the GPU job where serve_vlm.sh started it, and only once serve_vlm.sh says
# it's up. Inside a running job:
# srun --jobid=<job> --overlap bash infra/pace/stage2_cutin.sbatch 1
# For a whole list in one go, with the server started and stopped around it, see stage2_list.sh.
# VLM_CONFIG=<path> runs with another stage2.toml, as for trying another model. The default is
# configs/stage2.toml.
# Stage 2's outputs, under the segment's stage2/, describe a real scene, so they stay on PACE: Andrew
# reviews them in OnDemand, through the link $WOD_HOME/review/stage2/<prefix>_<rank>. Only counts print,
# with <prefix>_<rank> in place of the label, a hash of the name, and of the name itself.
set -euo pipefail
source infra/pace/env.sh
rank=${1:-}
model=${2:-dry-run}
list=${LIST:-cutin_top.txt}
if ! [[ $rank =~ ^[1-9][0-9]*$ && $model =~ ^(local|claude|dry-run)$ && $list =~ ^[a-z0-9_]+\.txt$ ]]; then
  echo "usage: [LIST=<list>.txt] stage2_cutin.sbatch <rank> [local|claude|dry-run]" >&2
  exit 2
fi
case $list in
  cutin_top.txt) scene=cutin_$rank ;;
  *) scene=${list%.txt}_$rank ;;
esac
if [ "$model" = local ] && ! bash infra/pace/serve_vlm.sh status; then  # its /health, from the server it started
  echo "$scene: no local model: start it first with infra/pace/serve_vlm.sh start" >&2
  exit 1
fi
context=$(sed -n "${rank}p" "$WOD_HOME/selected/$list")
if ! [[ $context =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "$scene: no such rank" >&2
  exit 2
fi
split=
for candidate in training validation; do
  if [ -f "$WOD_ROOT/$candidate/camera_image/$context.parquet" ]; then
    split=$candidate
  fi
done
if [ -z "$split" ]; then
  echo "$scene: no camera data" >&2
  exit 1
fi
if [ "$model" = claude ]; then
  source "$HOME/.config/agatha/anthropic.env"  # exports the key, for agatha-vlm only. Never echo it.
fi
status=0
uv run --frozen --extra claude agatha-vlm --config configs/stage1.toml --vlm-config "${VLM_CONFIG:-configs/stage2.toml}" \
  --split "$split" --context "$context" --model "$model" 2>&1 \
  | sed -E -e "s/sha256:[0-9a-f]{12}/$scene/g" -e "s/$context/$scene/g" || status=$?
echo "$scene: agatha-vlm exit $status"
if [ "$status" = 0 ]; then  # the results by scene, as the review pages are
  mkdir -p "$WOD_HOME/review/stage2"
  ln -sfn "$WOD_OUTPUTS/$context/stage2" "$WOD_HOME/review/stage2/$scene"
fi

#!/bin/bash
# usage: sbatch infra/pace/stage3_ingest.sbatch [cutin_top.txt random15.txt]
# CPU only. Source results remain read-only; new clips go under $WOD_HOME/stage3.
# A separate environment avoids changing the running Stage 1/2 environment.
#SBATCH --job-name=agatha-stage3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$HOME/agatha"
source infra/pace/env.sh
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/venvs/agatha-stage3"
set -a
source "$HOME/.config/agatha/supabase.env"
set +a
if [ "$#" -eq 0 ]; then set -- cutin_top.txt random15.txt; fi
for list in "$@"; do
  if ! [[ "$list" =~ ^[a-z0-9_]+\.txt$ ]]; then
    echo 'Stage 3: invalid scene list' >&2
    exit 2
  fi
  labels=()
  prefix=${list%.txt}
  if [ "$list" = random15.txt ]; then
    labels=(--human-labels "$WOD_HOME/review/$prefix/labels")
  fi
  uv run --frozen --extra stage3 agatha-stage3 ingest \
    --list "$WOD_HOME/selected/$list" --outputs "$WOD_OUTPUTS" \
    --clips "$WOD_HOME/stage3/clips" "${labels[@]}"
done

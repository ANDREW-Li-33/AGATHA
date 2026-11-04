#!/usr/bin/env bash
# Run on ICE: submit a job from the repo with its log in the private directory.
# infra/pace/submit.sh <job>.sbatch [args...]
set -euo pipefail
cd "$HOME/agatha"
job=$1
shift
exec sbatch --parsable -o "$HOME/scratch/waymo/logs/%x-%j.out" "infra/pace/$job" "$@"

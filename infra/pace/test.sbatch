#!/bin/bash
#SBATCH -J stage1-test
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 0:30:00
# The synthetic suite only, so full tracebacks are safe here.
set -euo pipefail
source infra/pace/env.sh
uv run --frozen pytest -q

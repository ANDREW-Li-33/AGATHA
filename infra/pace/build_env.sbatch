#!/bin/bash
#SBATCH -J build-env
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH -t 0:30:00
# PACE asks that installs run on compute nodes. --frozen means the same lock as the Mac.
set -euo pipefail
source infra/pace/env.sh
uv sync --frozen --python 3.12
uv run --frozen python -c "import cv2, numpy, pandas, pyarrow; print(numpy.__version__, pandas.__version__, pyarrow.__version__, cv2.__version__)"

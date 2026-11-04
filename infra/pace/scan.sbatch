#!/bin/bash
#SBATCH -J cutin-scan
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH -t 1:00:00
# Run order: download_labels.sbatch, then scan.sbatch, then download_listed.sbatch for each list
# (cutin_top.txt, fast_left.txt), then review_cutins.sbatch for each list.
# The label-only cut-in scan over every labeled segment, once download_labels.sbatch has run.
# Real data: the CLI prints only counts and the report's path. Fetch the report with pull_report.sh
# scan. Don't read events.parquet, segments.parquet, cutin_top.txt or fast_left.txt: they name contexts.
set -euo pipefail
source infra/pace/env.sh
uv run --frozen agatha-scan --config configs/stage1.toml --scan-config configs/scan.toml --workers 8 --top 10

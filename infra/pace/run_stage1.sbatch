#!/bin/bash
#SBATCH -J stage1
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 1:00:00
# usage: run_stage1.sbatch <split> <context_name> [extra agatha-stage1 flags]
# Real data: the CLI prints only counts, and failures record type and file:line.
# Fetch the report with pull_report.sh. Don't read this job's raw output, and don't list
# $WOD_OUTPUTS: its directory names are the context list.
set -euo pipefail
source infra/pace/env.sh
split=$1
context=$2
shift 2
uv run --frozen agatha-stage1 --config configs/stage1.toml --split "$split" --context "$context" "$@"

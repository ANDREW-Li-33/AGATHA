#!/usr/bin/env bash
# Fetch one report from ICE by its exact label, the only Dataset-derived thing that leaves.
# Reports are fetched by name, never listed: a list of labels is a context list, because a
# label is an unsalted hash of a name that anyone with the bucket listing can invert.
# usage: infra/pace/pull_report.sh <label>
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p reports
scp -q -F infra/pace/ssh_config "ice:scratch/waymo/outputs/reports/$1.json" reports/
echo "reports/$1.json"

#!/usr/bin/env bash
# Copy the code to ICE. Code only: data and outputs never travel in this direction.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -az --delete -e "ssh -F infra/pace/ssh_config" \
  --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' \
  --exclude docs/research --exclude data --exclude outputs --exclude reports --exclude scenario-miner \
  ./ ice:agatha/

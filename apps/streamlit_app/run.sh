#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --frozen --extra stage3 streamlit run apps/streamlit_app/app.py \
  --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

#!/usr/bin/env bash
# Stream Waymo Perception v1.4.3 segments: download one TFRecord (~1 GB), extract
# FRONT-camera samples (~1 MB each), delete the TFRecord, repeat. Resumable.
#
# Prereqs (one time, done by you):
#   1. Register at https://waymo.com/open and accept the dataset license with your Google account.
#   2. Install the gcloud CLI and run `gcloud auth login` with that same account.
#
# Usage: scripts/fetch_and_extract.sh [N_SEGMENTS=30]   (env: SPLIT=validation|training, EVERY_N=20)
set -euo pipefail

N=${1:-30}
SPLIT=${SPLIT:-validation}
EVERY_N=${EVERY_N:-20}
BUCKET="gs://waymo_open_dataset_v_1_4_3/individual_files/${SPLIT}"
RAW=data/raw
mkdir -p "$RAW" data/extracted

# Deterministic random sample of segments (not just the alphabetically first N). The
# order is stable, so a larger N extends the list and finished segments are skipped.
# awk, not head: head exits early, and SIGPIPE upstream would trip pipefail.
gcloud storage ls "${BUCKET}/*.tfrecord" \
  | awk 'BEGIN { srand(42) } { print rand() "\t" $0 }' | sort -n | cut -f2 \
  | awk -v n="$N" 'NR <= n' > data/segments.txt
if [ ! -s data/segments.txt ]; then
  echo "No files under ${BUCKET}. Check the bucket layout in the Cloud console and update BUCKET." >&2
  exit 1
fi

while read -r uri; do
  name=$(basename "$uri")
  [ -e "data/raw/.done_${name}" ] && continue
  echo "== ${name}"
  gcloud storage cp "$uri" "${RAW}/${name}"
  .venv/bin/miner extract "${RAW}/${name}" --every-n "$EVERY_N"
  rm -f "${RAW}/${name}"
  touch "data/raw/.done_${name}"
done < data/segments.txt

echo "Done. Next: .venv/bin/miner ingest && .venv/bin/miner split && .venv/bin/miner render --random 12"

#!/bin/bash
#SBATCH -J wod-listed
#SBATCH -N1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH -t 4:00:00
# usage: download_listed.sbatch cutin_top.txt|fast_left.txt
# Run order: download_labels.sbatch, then scan.sbatch, then download_listed.sbatch for each list
# (cutin_top.txt, fast_left.txt), then review_cutins.sbatch for each list.
# The camera components Stage 1 needs for each segment in a list the scan wrote. The labels and poses
# are here already, and the split is whichever holds the labels. One gcloud storage invocation at a
# time: it doesn't support parallel runs on a node. Lines print by number, never by name, so gcloud's
# own output goes nowhere: its errors name the object. A failure doesn't stop the rest.
# Up to 20 segments, twice --top. A single-segment download job took 16 s on ICE, so 4 hours
# leaves wide room.
set -euo pipefail
source infra/pace/env.sh
case ${1:-} in
  cutin_top.txt | fast_left.txt) ;;
  *) echo "usage: download_listed.sbatch cutin_top.txt|fast_left.txt" >&2; exit 2 ;;
esac
bucket=gs://waymo_open_dataset_v_2_0_1
line=0
# The list comes in on descriptor 3, so nothing that reads stdin can eat its lines.
while IFS= read -r context <&3; do
  line=$((line + 1))
  status=1  # no split holds its labels
  for split in training validation; do
    if [ -f "$WOD_ROOT/$split/lidar_box/$context.parquet" ]; then
      status=0
      for component in camera_image camera_calibration lidar_camera_synced_box projected_lidar_box; do
        mkdir -p "$WOD_ROOT/$split/$component"
        gcloud storage cp --no-user-output-enabled "$bucket/$split/$component/$context.parquet" \
          "$WOD_ROOT/$split/$component/" > /dev/null 2>&1 || status=$?
      done
    fi
  done
  echo "line $line: exit $status"
done 3< "$WOD_HOME/selected/$1"
du -sh "$WOD_ROOT" | cut -f1

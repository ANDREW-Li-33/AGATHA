#!/bin/bash
#SBATCH -J wod-labels
#SBATCH -N1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH -t 1:00:00
# Run order: download_labels.sbatch, then scan.sbatch, then download_listed.sbatch for each list
# (cutin_top.txt, fast_left.txt), then review_cutins.sbatch for each list.
# Labels, poses, stats and camera calibration for every labeled segment: the cut-in scan's input, a few GB.
# The calibration is one row per camera, so it's tiny, and the scan needs FRONT_LEFT's to count boxes.
# gcloud storage rsync exits non-zero on the bucket's _metadata entry after copying every
# file, so its exit code is ignored and the Parquet files are counted instead. Its own output
# goes nowhere: its errors name the object.
set -uo pipefail
source infra/pace/env.sh
bucket=gs://waymo_open_dataset_v_2_0_1
for split in training validation; do
  for component in lidar_box vehicle_pose stats camera_calibration; do
    mkdir -p "$WOD_ROOT/$split/$component"
    gcloud storage rsync --no-user-output-enabled --recursive \
      "$bucket/$split/$component" "$WOD_ROOT/$split/$component" > /dev/null 2>&1 || true
    printf '%s %s %s\n' "$split" "$component" "$(ls "$WOD_ROOT/$split/$component" | grep -c '\.parquet$')"
  done
done
du -sh "$WOD_ROOT" | cut -f1

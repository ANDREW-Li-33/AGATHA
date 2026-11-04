#!/bin/bash
#SBATCH -J wod-download
#SBATCH -N1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH -t 0:30:00
# usage: download_segment.sbatch <split> <context_name>
# One gcloud storage invocation at a time: it doesn't support parallel runs on a node.
set -euo pipefail
source infra/pace/env.sh
split=$1
context=$2
bucket=gs://waymo_open_dataset_v_2_0_1
for component in camera_image camera_calibration vehicle_pose lidar_box \
    lidar_camera_synced_box projected_lidar_box; do
  mkdir -p "$WOD_ROOT/$split/$component"
  gcloud storage cp --no-user-output-enabled \
    "$bucket/$split/$component/$context.parquet" "$WOD_ROOT/$split/$component/"
done
du -sh "$WOD_ROOT/$split" | cut -f1

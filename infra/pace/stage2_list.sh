#!/bin/bash
# usage: stage2_list.sh <list>.txt [local|claude]
# Stage 2 on every scene of $WOD_HOME/selected/<list>.txt in turn, each through stage2_cutin.sbatch with
# LIST set: it prints the scene's counts as <prefix>_<rank> and links its results to
# $WOD_HOME/review/stage2/<prefix>_<rank>. Run it after review_cutins.sbatch has run Stage 1 on the list.
# local, the default, asks Gemma 4 on this node: the script starts serve_vlm.sh first and stops it on
# exit, however the script ends, so the GPU is free again. A server that was up already is used, then
# stopped too. claude sends to Claude's API, which costs money, so it needs Andrew's go-ahead.
# RANKS="3 7" runs only those ranks. VLM_CONFIG=<path> reaches serve_vlm.sh and agatha-vlm alike, as
# for a config with a higher image budget.
# Run it inside a GPU job. An srun started from an SSH session ends when the connection drops, so launch
# it detached, from ~/agatha on the login node. env.sh sets $WOD_HOME, which a fresh login lacks:
#   source infra/pace/env.sh; nohup setsid srun --jobid=<job> --overlap bash -l infra/pace/stage2_list.sh random15.txt local > $WOD_HOME/logs/stage2-random15.log 2>&1 &
# Each scene's lines reach the log as the scene prints them. A scene that fails doesn't stop the rest.
# A line at the end counts the scenes run and the failures, naming their scenes, and the exit status is
# 1 if any failed.
set -euo pipefail
source infra/pace/env.sh
list=${1:-}
model=${2:-local}
if ! [[ $list =~ ^[a-z0-9_]+\.txt$ && $model =~ ^(local|claude)$ && -f $WOD_HOME/selected/$list ]]; then
  echo "usage: stage2_list.sh <list>.txt [local|claude], a list in \$WOD_HOME/selected" >&2
  exit 2
fi
case $list in
  cutin_top.txt) prefix=cutin ;;
  *) prefix=${list%.txt} ;;
esac
# Every rank is checked before the server starts, which takes minutes of the GPU's time.
lines=$(awk 'END {print NR}' "$WOD_HOME/selected/$list")
ranks=${RANKS:-$(awk '{print NR}' "$WOD_HOME/selected/$list")}  # every line, by number
count=$(wc -w <<< "$ranks" | tr -d ' ')
for rank in $ranks; do
  if ! [[ $rank =~ ^[1-9][0-9]*$ ]] || ! [ "$rank" -le "$lines" ]; then
    echo "stage2_list.sh: RANKS takes line numbers of $list, 1 to $lines, such as RANKS=\"3 7\"" >&2
    exit 2
  fi
done
if [ "$count" = 0 ]; then
  echo "stage2_list.sh: no scenes to run in $list" >&2
  exit 2
fi
out=$(mktemp)  # the running scene's lines, as they reach the log, for its last line
trap 'rm -f "$out"' EXIT
trap 'exit 1' HUP INT TERM  # through the EXIT trap, however this script is stopped
if [ "$model" = local ]; then
  trap 'rm -f "$out"; bash infra/pace/serve_vlm.sh stop' EXIT  # set before start, so a stop midway stops it too
  bash infra/pace/serve_vlm.sh start
fi
failed=
for rank in $ranks; do
  # stage2_cutin.sbatch exits 0 once agatha-vlm has run, however that went, so its last line says.
  status=0
  LIST=$list bash infra/pace/stage2_cutin.sbatch "$rank" "$model" | tee "$out" || status=$?  # pipefail: its status
  if [ "$status" != 0 ] || [[ $(tail -n 1 "$out") != *"agatha-vlm exit 0" ]]; then
    failed="$failed ${prefix}_$rank"
  fi
done
echo "$prefix: $count run, $(wc -w <<< "$failed" | tr -d ' ') failed${failed:+:$failed}"
[ -z "$failed" ]

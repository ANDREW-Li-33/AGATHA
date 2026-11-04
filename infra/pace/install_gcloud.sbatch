#!/bin/bash
#SBATCH -J install-gcloud
#SBATCH -N1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH -t 0:20:00
# The tarball bundles its own Python. install.sh is skipped so nothing edits
# ~/.bashrc: PACE warns against shell-init changes. env.sh adds it to PATH instead.
set -euo pipefail
cd "$HOME"
tarball=google-cloud-cli-linux-x86_64.tar.gz
curl -fsSLO "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/$tarball"
du -h "$tarball" | cut -f1
tar -xf "$tarball"
rm "$tarball"
"$HOME/google-cloud-sdk/bin/gcloud" version
du -sh "$HOME/google-cloud-sdk" | cut -f1

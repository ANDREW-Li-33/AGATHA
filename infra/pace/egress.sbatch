#!/bin/bash
#SBATCH -J egress
#SBATCH -N1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH -t 0:10:00
# Can a compute node reach everything setup needs? Any HTTP status means reachable.
hostname
for url in https://storage.googleapis.com/ https://oauth2.googleapis.com/token \
    https://accounts.google.com/ https://dl.google.com/ https://pypi.org/simple/ \
    https://files.pythonhosted.org/ https://github.com/ https://releases.astral.sh/; do
  printf '%s ' "$url"
  curl -sS -o /dev/null -m 20 -w '%{http_code} %{remote_ip}\n' "$url" || echo FAIL
done
source infra/pace/env.sh
uv --version
uv python list 3.12 --only-installed || true

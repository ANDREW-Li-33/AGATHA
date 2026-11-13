#!/bin/bash
#SBATCH -J build-vllm
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH -t 2:00:00
# usage: build_vllm.sbatch [qat]
# vLLM for agatha-vlm --model local, in its own environment, ~/scratch/venvs/vllm: it pins its own PyTorch
# and CUDA libraries, which AGATHA's lock knows nothing of, so AGATHA's environment is never touched. Then
# Gemma 4 31B's BF16 weights into $HF_HOME on scratch, 62.6 GB, or with qat, Google's 4-bit QAT checkpoint
# for vLLM instead, 23.3 GB: run it twice for both. Neither is gated, so no sign-in is needed. It downloads
# only when scratch's quota has room, as pace-quota reports it: a download the quota stops leaves partial
# files that still count against it, and a full scratch stops every other job's writes.
# PyPI's vLLM 0.32.0 is built for CUDA 13.0, which ICE's GPU nodes ran on 2026-11-04. CUDA=129 installs
# vLLM's own CUDA 12.9 wheel instead, for a driver older than R580: nvidia-smi on a GPU node shows it.
set -euo pipefail
source infra/pace/env.sh
case ${1:-} in
  "" | qat) ;;
  *) echo "usage: build_vllm.sbatch [qat]" >&2; exit 2 ;;
esac
version=0.32.0  # Gemma 4 needs 0.19.1 or later. 0.32.0 (2026-10-18) is the newest release over two weeks old.
venv=$HOME/scratch/venvs/vllm
cuda=${CUDA:-130}
if [ "$cuda" = 130 ]; then
  wheel="vllm==$version"
else
  wheel="https://github.com/vllm-project/vllm/releases/download/v$version/vllm-$version+cu$cuda-cp38-abi3-manylinux_2_28_x86_64.whl"
fi
uv venv --allow-existing --seed --python 3.12 "$venv"
# PyTorch for the same CUDA as vLLM's wheel. Named, not auto: auto reads the driver, and a CPU node has none.
uv pip install --python "$venv/bin/python" --torch-backend "cu$cuda" "$wheel"
status=0
"$venv/bin/vllm" download-kernels > /dev/null 2>&1 || status=$?  # FlashInfer's prebuilt kernels, if it has them
echo "vllm download-kernels: exit $status"
"$venv/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
model=google/gemma-4-31B-it
if [ "${1:-}" = qat ]; then
  model=google/gemma-4-31B-it-qat-w4a16-ct
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
# What's left to download, in GB, rounded up: nothing once it's all on scratch.
need=$("$venv/bin/python" -c "import math, sys; from huggingface_hub import snapshot_download
files = snapshot_download(sys.argv[1], dry_run=True)
print(math.ceil(sum(file.file_size for file in files if file.will_download) / 1e9))" "$model")
# Scratch's room under its quota, in GB: pace-quota's Scratch line gives the use, then the limit.
free=$(pace-quota 2> /dev/null | awk '$1 ~ /^Scratch:/ {print int($3 - $2)}' || true)
if ! [[ $need =~ ^[0-9]+$ && $free =~ ^[0-9]+$ ]]; then
  echo "$model: not downloaded: no size from the Hub, or no Scratch line from pace-quota" >&2
  exit 1
fi
echo "$model: $need GB to download, $free GB free on scratch"
if [ "$need" -gt "$free" ]; then
  echo "$model: not downloaded: free space on scratch first" >&2
  exit 1
fi
"$venv/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" \
  "$model" > /dev/null
echo "$model: downloaded"
du -sh "$HF_HOME" | cut -f1

# Sourced by every job. Keeps caches, the environment, and everything derived
# from the Dataset on scratch: home is small and has daily snapshots, which
# would work against deleting Dataset files promptly.
umask 077
export WOD_HOME="$HOME/scratch/waymo"
export WOD_ROOT="$WOD_HOME/v2_0_1"
export WOD_OUTPUTS="$WOD_HOME/outputs"
export UV_CACHE_DIR="$HOME/scratch/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$HOME/scratch/.uv_python"
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/venvs/agatha"
export HF_HOME="$HOME/scratch/hf"             # Gemma 4's weights, for build_vllm.sbatch and serve_vlm.sh
export XDG_CACHE_HOME="$HOME/scratch/.cache"  # vLLM's compiled kernels, among others
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
module load uv/0.9.17

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="${LOCALLLM_RUNTIME:-/mnt/d/local-llm-runtime}"
export HF_HOME="$runtime/huggingface"
export UV_CACHE_DIR="$runtime/cache"
export TMPDIR="/tmp/local-llm"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="$runtime/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor"
export TRITON_CACHE_DIR="$runtime/triton"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PURE_TRL_MODE=1
export WANDB_MODE=disabled
if [[ "${LOCALLLM_ENV:-train}" == serve && -x "$runtime/cuda-12.8/bin/nvcc" ]]; then
    export CUDA_HOME="$runtime/cuda-12.8"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
overlay="$HOME/.cache/local-llm/native-packages/${LOCALLLM_ENV:-train}/site-packages"
if [[ -f "$overlay/.ready" ]]; then
    export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
fi
export PATH="$runtime/${LOCALLLM_ENV:-train}/bin:$PATH"
exec python "$@"

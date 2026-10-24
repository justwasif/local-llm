#!/usr/bin/env bash
# Keep large environments/caches on D:, which has room for CUDA and weights.
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCALLLM_RUNTIME="${LOCALLLM_RUNTIME:-/mnt/d/local-llm-runtime}"
mkdir -p "$LOCALLLM_RUNTIME/tools" "$LOCALLLM_RUNTIME/cache" "$LOCALLLM_RUNTIME/huggingface"
mkdir -p outputs results/benchmarks results/safety results/reports logs
export UV_CACHE_DIR="$LOCALLLM_RUNTIME/cache"
export HF_HOME="$LOCALLLM_RUNTIME/huggingface"
python3 -m pip install --target "$LOCALLLM_RUNTIME/tools" uv
"$LOCALLLM_RUNTIME/tools/bin/uv" venv --python python3 "$LOCALLLM_RUNTIME/train"
"$LOCALLLM_RUNTIME/tools/bin/uv" venv --python python3 "$LOCALLLM_RUNTIME/serve"
echo "Runtime prepared at $LOCALLLM_RUNTIME"

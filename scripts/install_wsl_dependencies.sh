#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="${LOCALLLM_RUNTIME:-/mnt/d/local-llm-runtime}"
export UV_CACHE_DIR="$runtime/cache"
export TMPDIR="$runtime/tmp"
mkdir -p "$TMPDIR" logs
target="${1:-train}"
shift || true
"$runtime/tools/bin/uv" pip install --python "$runtime/$target/bin/python" --link-mode copy -r "configs/requirements-$target.txt" "$@"

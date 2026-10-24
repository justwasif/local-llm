#!/usr/bin/env bash
set -euo pipefail
runtime="${LOCALLLM_RUNTIME:-/mnt/d/local-llm-runtime}"
cuda="$runtime/cuda-12.8"
packages="$runtime/serve/lib/python3.12/site-packages/nvidia"
chmod +x "$cuda/bin/nvcc" "$cuda/bin/ptxas" "$cuda/bin/fatbinary" "$cuda/bin/nvlink" "$cuda/nvvm/bin/cicc"
ln -sfn "$packages/cuda_runtime/lib" "$cuda/lib64"
ln -sfn "$packages/cuda_runtime/lib/libcudart.so.12" "$packages/cuda_runtime/lib/libcudart.so"
for header in "$packages/cuda_runtime/include/"*; do
  [[ -e "$cuda/include/$(basename "$header")" ]] || ln -s "$header" "$cuda/include/$(basename "$header")"
done
if [[ -d "$packages/cuda_cccl/include" ]]; then
  for header in "$packages/cuda_cccl/include/"*; do
    [[ -e "$cuda/include/$(basename "$header")" ]] || ln -s "$header" "$cuda/include/$(basename "$header")"
  done
fi
"$cuda/bin/nvcc" --version

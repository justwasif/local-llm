# Stage 07 — Deployment

Implemented `configs/sglang/server_config.yaml` and `scripts/serve/launch_server.py`. On this Nix host launch through `scripts/with_gpu_runtime.sh .venv/bin/python scripts/serve/launch_server.py`.

SGLang has not successfully served a model yet. The latest local launch was blocked during SGLang/Torch kernel initialization (`torch._C._cpu._is_amx_tile_supported` missing) on this CUDA 12.8 driver environment. Do not churn package versions on the nearly-full disk; resolve this on a compatible remote GPU/runtime or after a deliberate environment rebuild.

Measure throughput, latency, model quantization, and VRAM after serving on target hardware.

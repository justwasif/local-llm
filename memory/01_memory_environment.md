# Stage 01 - Environment

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

Installed separate training (291 packages) and serving environments. Both passed uv pip check. verify_setup.py --full passed on the current GPU.

## File manifest

scripts/setup_wsl.sh; scripts/install_wsl_dependencies.sh; scripts/wsl_run.sh; scripts/prepare_native_overlay.py; scripts/cache_package_metadata.py; scripts/install_cuda_compiler.py; scripts/link_cuda_compiler.sh; configs/requirements-*.txt and *.lock.txt.

## How it works

Large environments, model files and caches live at D:/local-llm-runtime. Python-heavy packages are mirrored into /home/krish/.cache/local-llm/native-packages/<environment>/site-packages and linked to the same compiled libraries. The literal site-packages path is required for dill/Hugging Face dataset fingerprinting. The wrapper activates a cache only after its .ready marker exists; regenerate after dependency changes.

## Configuration

Ubuntu Python 3.12.3; RTX 4070 Laptop GPU, 8188 MiB VRAM. Train: AlignTune 0.1.28, Torch 2.11.0, Transformers 5.20.2, TRL 1.7.1, PEFT 0.21.2, bitsandbytes 0.50.2, CuratorKIT 0.1.22, AuditKIT 1.1.3, Gradio 6.30.0. Serve: SGLang 0.5.9, Torch 2.9.1, Transformers 4.57.1; verified NVIDIA CUDA compiler 12.8.93 and CCCL headers.

## Dependencies

CUDA driver must be visible in WSL. The two environments intentionally cannot be combined.

## Troubleshooting

Windows Python alias is unusable; host helper scripts ran with C:/Users/Krish/miniconda3/python.exe. Use scripts/wsl_run.sh for training. Do not install SGLang into the training environment. Native overlay symlinks require the D: environments to remain present. /tmp/local-llm is used for POSIX temporary-file semantics.

## Verification

Run wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/wsl_run.sh scripts/verify_setup.py --full; uv pip check --python <environment>/bin/python. Package presence alone is not workflow completion.

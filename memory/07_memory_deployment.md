# Stage 07 - SGLang serving

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

SGLang completed GPU warm-up and real generation for each model with a completed evaluation artifact. Export checks compare vocabulary, 12 language samples, all 86 evaluation prompts, chat-template token IDs, model architecture, RoPE and dtype. Inspect results/reports/{sft,dpo}_export_verification.json and the model path recorded by each evaluation.

## File manifest

scripts/serve/local_server.py; scripts/serve/launch_server.py; results/reports/model_download.json; scripts/link_cuda_compiler.sh

## How it works

Use the current WSL wrapper and record artifacts from successful real executions. Preserve failures and negative deltas.

## Configuration

Local endpoint 127.0.0.1:30000, GUI 127.0.0.1:7860. Base and candidate evaluation must use the same frozen prompts and generation settings.

## Dependencies

Depends on completed preceding stages and the current machine's artifacts.

## Troubleshooting

SGLang RoPE kernels require the full matching nvcc archive, runtime headers and CCCL; the pip nvcc package alone only provides ptxas. Original handoff's AMX error does not describe this rebuilt environment.

## Verification

Inspect actual reports, artifacts and live API/UI behavior before marking the stage finished.

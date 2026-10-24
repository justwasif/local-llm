# Stage 09 - Benchmark report

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

The report generator writes only available measured artifacts. No missing training/evaluation stage is marked completed. Submission readiness requires representative benchmark gains and independent safety/native-speaker review.

## File manifest

scripts/write_run_report.py; results/reports/FINAL_BENCHMARK_REPORT.md

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

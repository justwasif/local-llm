# LocalLLM current handoff

Updated 2026-10-25. This is the Windows/Ubuntu WSL2 continuation, replacing the previous Nix-host assumptions.

## Verified

- Both dependency environments are installed and compatible; GPU is RTX 4070 Laptop (8 GB).
- Pinned base 1.5B model and source/evaluation data are downloaded with provenance. CuratorKIT cleaning, QA, multi-turn and preference generation completed; source-bound review excluded two factual errors. Training data contain 220 SFT conversations and seven reviewed preference pairs.
- SGLang passed real GPU warm-up and generation after installing matching CUDA compiler/runtime/CCCL headers.
- Full project suite passes: 25 tests. Ruff and Python compile checks also pass; additional toolkit integration tests validate generation inputs.

## Continue by evidence

- SFT: completed, 55 optimizer steps
- DPO: completed, 2 optimizer steps
- baseline evaluation: completed; 86 quality predictions and six safety probes; mean ten-language FLORES pilot chrF++ 9.8761; lexical safety score 0.5000
- sft evaluation: completed; 86 quality predictions and six safety probes; mean ten-language FLORES pilot chrF++ 8.7548; lexical safety score 0.3333
- dpo evaluation: completed; 86 quality predictions and six safety probes; mean ten-language FLORES pilot chrF++ 8.9250; lexical safety score 0.3333
- Final technical verification: passed; inspect results/reports/final_verification.json for command output and artifact-integrity checks.
- Live GUI/API: passed, http://127.0.0.1:7860. The earlier visual-check block is resolved: Taste Skill screenshots and responsive measurements are recorded in `.taste-skill/after-final/` at 1440, 768 and 390 pixels, with no inspector findings.
- Checkpoint-specific AuditKIT caching includes weight SHA-256 hashes. Preserve actual negative benchmark deltas; shared API model names alone are insufficient cache identity.

## Commands

Use `wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/wsl_run.sh <script>` for train/tools. Prepend `env LOCALLLM_ENV=serve` after `--` for serving. See README.md for stage commands. Never run GPU training while SGLang occupies the GPU.

## Remaining project targets

The 7B target needs 16-24 GB VRAM. Full standard benchmarks, 10+ language/code-mixed training coverage, independent safety and native-speaker review, and representative benchmark gains remain separate from executing the small local pipeline. Do not claim these without new evidence.
The PDF's PS4 outcome requires improvement, not merely executed training. Consult results/reports/FINAL_BENCHMARK_REPORT.md before making that claim; this pilot is too small for statistical superiority and any regression must remain visible. The PDF itself provides no numeric benchmark thresholds.

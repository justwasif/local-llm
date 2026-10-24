# LocalLLM — Remaining Work Handoff

Last updated: 2026-10-24

## Current verified state

- Project structure, configurations, pipeline scripts, Gradio UI, and stage memory files exist.
- `.venv` has the core demo stack plus the required toolkit imports: `curatorkit==0.1.22`, `aligntune==0.1.28`, `auditkit==1.1.3`, and `sglang==0.4.10`.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6.06 GB VRAM.
- Free disk space is approximately 4.7 GB (97% full). Avoid large model downloads and package experiments.
- A real QLoRA smoke run succeeded using `Qwen/Qwen2.5-1.5B-Instruct` and eight reviewed demo examples. Its adapter is at `outputs/demo_1_5b_lora/`.
- The smoke model loads and generates Hindi, Bengali, and Hinglish output, but quality is poor due to the tiny dataset. Do not present it as a benchmark result or final model.
- GUI construction was tested successfully. It has not been exercised against a running SGLang endpoint.

## Host-specific runtime requirement

This is a Nix host. Python wheels need NVIDIA and C++ shared-library paths **before Python starts**.

Always launch GPU/native Python commands through (the wrapper now includes the Nix zlib path too):

```bash
scripts/with_gpu_runtime.sh .venv/bin/python <script-or-module>
```

Examples:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/verify_setup.py
scripts/with_gpu_runtime.sh .venv/bin/python scripts/training/run_demo_sft.py
scripts/with_gpu_runtime.sh .venv/bin/python src/gui/app.py
```

## Hackathon compliance gaps

The Local LLM problem-statement document requires:

1. SGLang as the core serving, inference, evaluation, or benchmarking engine.
2. CuratorKIT during data curation.
3. AlignTune during post-training/fine-tuning.
4. AuditKIT for evaluations and safety.

All four required packages are installed, but package presence alone does not establish compliance. CuratorKIT, AlignTune, and AuditKIT have not completed a real workflow, and SGLang has not served a model.

## Required next work, in order

### 1. Do not modify the local Python stack without a deliberate rebuild plan

The local SGLang launch is currently blocked during kernel initialization by `AttributeError: module 'torch._C._cpu' has no attribute '_is_amx_tile_supported'` in this CUDA 12.8/Nix environment. The disk is almost full. Do not spend more time pinning or reinstalling SGLang/Torch packages locally. Use a compatible remote GPU/runtime for the required SGLang evidence, or first free disk and define a clean supported environment.

### 2. Reassess capacity before training

The intended 7B production model is not feasible for QLoRA training or reliable SGLang operation on 6 GB VRAM. Do not attempt it without freeing disk space and accepting likely out-of-memory failure.

Use Qwen2.5-1.5B for a local proof-of-pipeline, or move the 7B workload to a 16–24 GB GPU machine. Check capacity first:

```bash
df -h .
nvidia-smi
```

### 3. Acquire and validate meaningful data

Two capped public samples have been downloaded: 100 Alpaca Hindi records and 100 Aya records. The current language report has 100 `unknown` rows and broad non-Indic coverage, so it is not ready for the advertised Indic training claim. Filter, deduplicate, inspect response quality, and record licences/provenance before using it. Use this command only after capacity allows a larger curated dataset:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/data/download_datasets.py --limit 100
scripts/with_gpu_runtime.sh .venv/bin/python scripts/data/verify_data.py
```

Increase limits only after checking disk capacity. Ensure material coverage of Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, and Hinglish if that remains the target claim.

### 4. Run CuratorKIT through a working SGLang endpoint

Start an SGLang endpoint using a model the GPU can host. For local validation, use 1.5B instead of the 7B default:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/serve/launch_server.py --model-path Qwen/Qwen2.5-1.5B-Instruct
```

In another terminal:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/data/run_curation.py
scripts/with_gpu_runtime.sh .venv/bin/python scripts/data/merge_curated_data.py
scripts/with_gpu_runtime.sh .venv/bin/python scripts/training/format_sft_data.py
```

Verify that `data/curated/final_sft_train.jsonl`, `data/curated/sft_messages_train.jsonl`, and `data/preference/multilingual_pref.jsonl` contain valid, nontrivial data.

### 5. Run AlignTune SFT and DPO

Existing production configs point to Qwen2.5-7B. Make a clearly named 1.5B local-demo config rather than silently changing the production config. Then:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/training/run_sft.py
scripts/with_gpu_runtime.sh .venv/bin/python scripts/training/run_dpo.py
scripts/with_gpu_runtime.sh .venv/bin/python scripts/training/merge_lora.py
```

Record actual dataset counts, loss, duration, peak VRAM, adapter/model paths, and failures. DPO requires real valid `(prompt, chosen, rejected)` records; it must not be marked complete before they exist.

### 6. Serve and test the trained model with SGLang

Start SGLang using the actual merged model and test OpenAI-compatible chat completions. Then launch the GUI:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/serve/launch_server.py
scripts/with_gpu_runtime.sh .venv/bin/python src/gui/app.py
```

Test one query per target language/domain, Unicode rendering, failure behavior when the server is unavailable, and basic latency.

### 7. Run AuditKIT evaluation and safety checks

With the SGLang server running:

```bash
scripts/with_gpu_runtime.sh .venv/bin/python scripts/eval/run_evaluation.py
```

Save genuine benchmark and safety output under `results/benchmarks/` and `results/safety/`. Do not invent BLEU, ROUGE, accuracy, hallucination, latency, or throughput scores.

### 8. Finalize evidence and documentation

Update these only from actual artifacts:

- `results/reports/FINAL_BENCHMARK_REPORT.md`
- `README.md`
- `memory/01` through `memory/09`

Include installed toolkit versions, model ID/revision, data provenance/licences, exact train/eval commands, hardware, measured metrics, safety failures/mitigations, SGLang performance, and GUI screenshots or a demo capture.

## Documentation rule for the next agent

The environment, data, deployment, GUI, and this handoff notes were updated with this continuation pass. All remaining stage notes must distinguish **implemented scaffolding** from **executed stages**. After each real stage finishes, overwrite the corresponding note with verified values rather than planned values.

## Do not claim yet

- Compliance with the mandated SGLang / CuratorKIT / AlignTune / AuditKIT workflow.
- A trained 7B multilingual model.
- DPO alignment.
- Benchmark improvement over Qwen.
- Safety evaluation completion.
- Production-ready quality or throughput.

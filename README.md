# LocalLLM — Multilingual Indic Assistant

LocalLLM is a locally deployable Qwen2.5-7B pipeline for Indian public-facing use cases: education, healthcare, agriculture, and government services. It covers Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Hinglish, and Banglish.

## Architecture

`data collection → CuratorKIT curation → QLoRA SFT → DPO → merge → SGLang → Gradio`

The project follows the build order in [`architecture/00_MASTER_PLAN.md`](architecture/00_MASTER_PLAN.md). Large remote downloads, GPU training, and server launches are intentionally explicit commands; they are not performed during project setup.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
scripts/with_gpu_runtime.sh python scripts/verify_setup.py

# Create local, versioned multilingual seed records.
python scripts/data/create_seed_data.py

# Optionally acquire public datasets (use a conservative smoke limit first).
python scripts/data/download_datasets.py --limit 100
```

After authentication with Hugging Face and dataset acquisition:

```bash
# Terminal 1: use the base model for CuratorKIT generation.
python scripts/serve/launch_server.py --model-path Qwen/Qwen2.5-7B-Instruct

# Terminal 2: curate then prepare training data.
python scripts/data/run_curation.py
python scripts/data/merge_curated_data.py
python scripts/training/format_sft_data.py
python scripts/training/run_sft.py
python scripts/training/run_dpo.py
python scripts/training/merge_lora.py

# Serve the merged model and start the demo.
python scripts/serve/launch_server.py
python src/gui/app.py
```

The Gradio demo listens on `http://127.0.0.1:7860`. Override endpoints without code changes with `SGLANG_BASE_URL`, `SGLANG_MODEL`, `GRADIO_SERVER_NAME`, and `GRADIO_SERVER_PORT`.

### 6 GB GPU smoke-training path

The production 7B configuration needs at least 16 GB VRAM. For a 6 GB RTX 4050, this repository includes a verified Qwen2.5-1.5B QLoRA smoke profile:

```bash
source .venv/bin/activate
python scripts/training/create_demo_training_data.py
scripts/with_gpu_runtime.sh python scripts/training/run_demo_sft.py
scripts/with_gpu_runtime.sh python scripts/training/quick_test_demo.py
```

This uses eight reviewed examples solely to validate data → QLoRA → saved adapter → generation. It is deliberately too small to demonstrate quality; use curated data at meaningful scale before evaluating the model.

## Evaluation

With the merged model server running:

```bash
python scripts/eval/run_evaluation.py
```

Reports are written to `results/benchmarks/` and `results/safety/`. Do not claim benchmark improvements until those commands have produced real reports.

## Layout

- `configs/` — CuratorKIT, AlignTune, AuditKIT, and SGLang settings
- `scripts/` — reproducible commands for every pipeline stage
- `src/gui/app.py` — streaming Gradio public demo
- `data/` — ignored large data artifacts; local seeds remain reproducible
- `memory/` — stage handoff notes and run-state expectations

## Hardware and safety

The intended target is a CUDA-capable GPU with at least 16 GB VRAM (24 GB recommended). Healthcare answers are informational only; government-service guidance should be verified on official websites or at authorised offices.

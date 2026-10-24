# 🏗️ Master Build Plan — LocalLLM: Low-Resource & Multilingual Language Model

> **Problem Statement 4**: Improve a model for underserved languages, dialects or code-mixed text in a public-facing setting such as education, public services, agriculture or healthcare.

> **Hackathon**: SYNTAX ERROR 2026 | OCT 23–25

---

## 🎯 Project Goal

Build a **production-ready, locally-deployable multilingual LLM** that excels at low-resource Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, and code-mixed Hinglish) for **public-facing domains** — education, healthcare, agriculture, and government services.

The model must:
1. Understand and generate text in 10+ Indic languages + code-mixed text
2. Beat baseline benchmarks on IndicNLP, FLORES, IndicXTREME
3. Run locally on consumer-grade GPUs (≤ 24GB VRAM)
4. Serve via SGLang with an interactive GUI

---

## 🛠️ Technology Stack (Mandatory per PS)

| Component | Tool | Purpose |
|-----------|------|---------|
| **Inference & Serving** | [SGLang](https://docs.sglang.io/) | Core engine (REQUIRED) |
| **Data Curation** | [CuratorKIT](https://curatorkit.lexsi.ai) | Dataset preparation (uses SGLang internally) |
| **Fine-Tuning** | [AlignTune](https://aligntune.lexsi.ai) | SFT + DPO training (uses Unsloth/TRL) |
| **Evaluation** | [AuditKIT](https://auditkit.lexsi.ai) | Benchmarking & safety (uses SGLang) |
| **Base Model** | Qwen2.5-7B-Instruct | Best multilingual open model at this size |
| **GUI** | Gradio | Interactive demo interface |

---

## 📂 Architecture Files — Execution Order

> [!IMPORTANT]
> **Agent Instructions**: Execute these files **sequentially**. Complete one file fully before moving to the next. After completing each file, write a memory file in `memory/` documenting everything implemented.

| Order | File | Stage | Description | Est. Time |
|-------|------|-------|-------------|-----------|
| 1 | [`01_ENVIRONMENT_SETUP.md`](./01_ENVIRONMENT_SETUP.md) | Setup | Python env, dependencies, GPU check, project structure | 15 min |
| 2 | [`02_DATA_COLLECTION.md`](./02_DATA_COLLECTION.md) | Stage 1 | Download & organize multilingual datasets | 30 min |
| 3 | [`03_DATA_CURATION.md`](./03_DATA_CURATION.md) | Stage 1 | CuratorKIT: clean, filter, generate synthetic data | 45 min |
| 4 | [`04_MODEL_FINETUNING.md`](./04_MODEL_FINETUNING.md) | Stage 2 | AlignTune: SFT on curated multilingual data | 60 min |
| 5 | [`05_DPO_ALIGNMENT.md`](./05_DPO_ALIGNMENT.md) | Stage 2 | AlignTune: DPO preference alignment | 45 min |
| 6 | [`06_EVALUATION.md`](./06_EVALUATION.md) | Stage 3 | AuditKIT + SGLang: benchmarks & safety evals | 30 min |
| 7 | [`07_SGLANG_DEPLOYMENT.md`](./07_SGLANG_DEPLOYMENT.md) | Stage 3 | SGLang server setup & optimization | 20 min |
| 8 | [`08_GUI_APPLICATION.md`](./08_GUI_APPLICATION.md) | Stage 3 | Gradio GUI with multilingual support | 30 min |
| 9 | [`09_BENCHMARKING_REPORT.md`](./09_BENCHMARKING_REPORT.md) | Final | Generate comparison report & submission artifacts | 20 min |

---

## 📁 Expected Project Structure After Completion

```
local llm/
├── architecture/           # This folder — build instructions
│   ├── 00_MASTER_PLAN.md
│   ├── 01_ENVIRONMENT_SETUP.md
│   ├── ...
│   └── 09_BENCHMARKING_REPORT.md
├── memory/                 # Agent memory files (auto-generated after each stage)
│   ├── 01_memory_environment.md
│   ├── 02_memory_data_collection.md
│   ├── ...
│   └── 09_memory_benchmarking.md
├── data/                   # All datasets
│   ├── raw/                # Downloaded raw datasets
│   ├── curated/            # CuratorKIT output
│   ├── preference/         # DPO preference pairs
│   └── eval/               # Evaluation datasets
├── configs/                # All YAML configs
│   ├── curatorkit/
│   ├── aligntune/
│   ├── auditkit/
│   └── sglang/
├── scripts/                # Python scripts
│   ├── data/               # Data processing scripts
│   ├── training/           # Training launch scripts
│   ├── eval/               # Evaluation scripts
│   └── serve/              # Serving scripts
├── src/                    # Source code
│   └── gui/                # Gradio GUI application
├── outputs/                # Training outputs
│   ├── sft_model/          # SFT checkpoints
│   ├── dpo_model/          # DPO checkpoints
│   └── merged_model/       # Final merged model
├── results/                # Evaluation results
│   ├── benchmarks/
│   ├── safety/
│   └── reports/
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧠 Memory System — Agent Instructions

> [!IMPORTANT]
> After completing each architecture file, the agent **MUST** create a memory file at:
> `memory/XX_memory_<stage_name>.md`
>
> Each memory file must contain:
> 1. **What was implemented** — list of all files created/modified
> 2. **File manifest** — every code file with its path, purpose, and key functions
> 3. **How it works** — technical explanation of the pipeline/logic
> 4. **Configuration values** — all important config parameters used
> 5. **Dependencies** — what this stage depends on and what depends on it
> 6. **Troubleshooting** — common issues and how to fix them
> 7. **Verification** — how to verify this stage completed successfully

---

## 🏆 Winning Strategy

1. **Model Selection**: Qwen2.5-7B-Instruct — best multilingual model at 7B scale, supports 29 languages including major Indic languages
2. **Data Quality > Quantity**: Use CuratorKIT to generate HIGH-QUALITY synthetic data for underserved languages, then filter aggressively
3. **Two-Stage Training**: SFT for domain knowledge → DPO for output quality alignment
4. **Benchmarking Edge**: Target IndicXTREME, FLORES, and custom domain-specific evals to show measurable improvement
5. **Demo Impact**: Beautiful Gradio GUI showing real-time multilingual chat with language switching

---

## ⚡ Quick Start (For Agents)

```bash
# Start by reading and executing:
# architecture/01_ENVIRONMENT_SETUP.md

# After each file:
# 1. Verify all steps completed
# 2. Write memory file to memory/
# 3. Move to next architecture file
```

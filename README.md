# LocalLLM - multilingual Indic assistant

This project attempts **PS4: low-resource and multilingual languages** from the original [Local LLM problem statement](LocalLLM%20PS.pdf). It uses CuratorKIT for curation, AlignTune/TRL for post-training, SGLang for GPU inference, AuditKIT for measured evaluations, and Gradio for the demo.

The current machine is Windows with Ubuntu WSL2 and an **RTX 4070 Laptop GPU (8 GB)**. The executable local profile uses **Qwen2.5-1.5B-Instruct** with 4-bit LoRA. The separate 7B production plan requires a larger GPU and has not been executed here. Earlier Nix/RTX 4050 notes described a different machine; their ignored artifacts were absent from this checkout.

Read [the execution report](results/reports/FINAL_BENCHMARK_REPORT.md) and [the current handoff](memory/10_remaining_work_handoff.md) for completed stages, measured results, and outstanding targets. A configured stage is not evidence that it ran. Pilot scores do not establish production quality or full benchmark superiority.

## Environments

Training and serving require **separate environments** because their Torch/Transformers constraints conflict. Large packages and model weights are stored under `D:/local-llm-runtime`, where capacity was checked before installation. `requirements.txt` installs the training stack only; serving uses `configs/requirements-serve.txt`. Exact installed versions are in `configs/requirements-{train,serve}.lock.txt`.

On this machine, from PowerShell in the repository:

```powershell
wsl -d Ubuntu -u root -- apt-get update
wsl -d Ubuntu -u root -- apt-get install -y pkg-config libicu-dev build-essential python3-dev
wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/setup_wsl.sh
wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/install_wsl_dependencies.sh train
wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/install_wsl_dependencies.sh serve
```

The Windows `python` alias was unusable. Host-only helpers used `C:/Users/Krish/miniconda3/python.exe`:

```powershell
& C:/Users/Krish/miniconda3/python.exe scripts/download_local_model.py
& C:/Users/Krish/miniconda3/python.exe scripts/install_cuda_compiler.py
wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/link_cuda_compiler.sh
& C:/Users/Krish/miniconda3/python.exe scripts/cache_package_metadata.py
& C:/Users/Krish/miniconda3/python.exe scripts/prepare_native_overlay.py train
& C:/Users/Krish/miniconda3/python.exe scripts/prepare_native_overlay.py serve
```

The last helpers cache Python code on native WSL storage to avoid slow file scanning across DrvFS. Compiled libraries remain linked to D:. The cache must end in `site-packages` (the helper creates this layout); an arbitrary path causes dill/Hugging Face dataset fingerprinting failures. Regenerate these caches after changing dependencies. A cache is activated only after its `.ready` marker is written. Temporary files use native `/tmp/local-llm` for POSIX semantics.

The CUDA compiler helper downloads NVIDIA's published CUDA 12.8.1 nvcc archive and verifies its SHA-256. The serving environment also needs runtime and CCCL headers. Installing the pip nvcc package alone supplies `ptxas`, not the full compiler needed by SGLang's RoPE JIT kernel.

## Local workflow

All following commands run **inside Ubuntu WSL**, from `/mnt/c/Users/Krish/local-llm`. The wrapper selects training/tools by default; prefix `LOCALLLM_ENV=serve` for serving. `LOCALLLM_RUNTIME` can override the runtime directory on another Linux machine; the Windows cache helpers are specific to this host.

```bash
bash scripts/wsl_run.sh scripts/verify_setup.py --full
bash scripts/wsl_run.sh scripts/data/acquire_indic_subset.py
bash scripts/wsl_run.sh scripts/data/acquire_flores_eval.py
bash scripts/wsl_run.sh scripts/data/create_seed_data.py
```

Acquisition freezes 216 Aya training records, 36 validation records and 36 test records across nine available Indic languages, with dataset revision and Apache-2.0 licence recorded. The FLORES-200 file is an **evaluation-only 50-example pilot**: five fixed devtest sentences in each of ten languages, with archive checksum and CC-BY-SA-4.0 attribution. Cached successful Aya requests survive transient HTTP failures. Do not rerun acquisition in the middle of a model comparison or train on evaluation files.

In terminal 1, serve the pinned base model:

```bash
LOCALLLM_ENV=serve bash scripts/wsl_run.sh scripts/serve/local_server.py
```

Wait for the real warm-up generation and the ready message. In terminal 2:

```bash
bash scripts/wsl_run.sh scripts/data/run_curation.py --dry-run
bash scripts/wsl_run.sh scripts/data/run_curation.py
# Inspect accepted synthetic records and record source-bound review decisions.
# configs/synthetic_review.json records this run; changed exports require re-review.
bash scripts/wsl_run.sh scripts/data/review_synthetic_data.py
bash scripts/wsl_run.sh scripts/data/run_curation.py --only reviewed
bash scripts/wsl_run.sh scripts/data/merge_curated_data.py
bash scripts/wsl_run.sh scripts/training/format_sft_data.py
bash scripts/wsl_run.sh scripts/eval/run_evaluation.py --label baseline
```

CuratorKIT keeps manifests, provenance, checksums and rejected records per pipeline. Input schema gates run before generation. The merge reads only canonical accepted exports, deduplicates prompts, preserves available source metadata, and rejects an empty SFT corpus. The formatter retains complete alternating exchanges and removes unanswered trailing user turns. QA/multi-turn grounding failures remain rejected.

Agent review also caught two factual errors that passed the model judge; their prompt fingerprints and the reviewed export checksums are versioned in `configs/synthetic_review.json`. It is specific to this run, and merging changed exports fails until a new review is recorded. `record_current_review.py` reproduces the recorded decisions only when the known failure fragments still match; it is not a general automatic factual reviewer.

DPO chosen answers are anchored to the repository's handwritten references, paired with reviewed weaker model-generated rejected text, and re-curated through CuratorKIT. This produced seven small preference pairs rather than accepting weak generated chosen answers. Raw model outputs are preserved. Agent review is not independent native-speaker or domain-expert annotation.

**Stop SGLang with Ctrl+C before training** so serving and training do not compete for the GPU:

```bash
bash scripts/wsl_run.sh scripts/training/run_sft.py --local
bash scripts/wsl_run.sh scripts/training/merge_lora.py --local sft
LOCALLLM_ENV=serve bash scripts/wsl_run.sh scripts/serve/verify_export.py --stage sft
```

Serve the merged SFT model and measure the same frozen evaluations:

```bash
LOCALLLM_ENV=serve bash scripts/wsl_run.sh scripts/serve/local_server.py --model-path /mnt/c/Users/Krish/local-llm/outputs/local_1_5b_sft_merged
# Another terminal:
bash scripts/wsl_run.sh scripts/eval/run_evaluation.py --label sft
```

Stop serving again before DPO, then merge and evaluate:

```bash
bash scripts/wsl_run.sh scripts/training/run_dpo.py --local
bash scripts/wsl_run.sh scripts/training/merge_lora.py --local dpo
LOCALLLM_ENV=serve bash scripts/wsl_run.sh scripts/serve/verify_export.py --stage dpo
LOCALLLM_ENV=serve bash scripts/wsl_run.sh scripts/serve/local_server.py --model-path /mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo_merged
# Another terminal:
bash scripts/wsl_run.sh scripts/eval/run_evaluation.py --label dpo
bash scripts/wsl_run.sh src/gui/app.py
```

The GUI is at **http://127.0.0.1:7860**; SGLang is at **http://127.0.0.1:30000/v1**. `SGLANG_BASE_URL`, `SGLANG_MODEL`, `SGLANG_API_KEY`, `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` override endpoints. The local SGLang launcher binds only to loopback and serves the name `local-llm`. A language dropdown is a requested output language, not evidence of validated competence.

The GUI uses a responsive, framed-sheet layout with a compact welcome state, Indic-language prompt examples, and a streaming conversation panel. Styling lives in `src/gui/ui.css`; the Taste Skill review, screenshots and measurements are in `.taste-skill/`.

## Verification and evidence

```bash
bash scripts/wsl_run.sh -m pytest tests -q
# With the final model and GUI running, after all three evaluations:
bash scripts/wsl_run.sh scripts/verify_live_gui.py
bash scripts/wsl_run.sh scripts/final_verification.py
bash scripts/wsl_run.sh scripts/write_run_report.py
bash scripts/wsl_run.sh scripts/update_handoff_notes.py
```

Tests cover dataset normalization/deduplication, rejected-output exclusion, invalid JSONL diagnostics, prompt identity, distinct preferences, real CuratorKIT reader/schema contracts, GUI construction, Unicode streaming/history, empty input and server failures. Real model training, reload, merged-model inference and benchmark reports provide integration evidence beyond unit tests.

Training reports and final validation record versions, optimizer steps, loss history, elapsed time, peak allocated VRAM, model revision and dataset fingerprint. AuditKIT saves raw predictions, scores, provenance and measured request performance. Evaluation labels are checked against SGLang's actual loaded model path. Cache identities include SHA-256 hashes of the actual served weights, so different checkpoints at the same API name cannot share predictions. `--resume` additionally requires identical saved prompts, settings and checkpoint identity. FLORES pilot reporting uses SacreBLEU's FLORES-200 tokenizer for spBLEU and chrF++ with exact signatures. This is not a full FLORES/IndicXTREME/MMLU result.

The merge helper adds compatible legacy RoPE/dtype fields and tokenizer special-token metadata for Transformers 4.x serving after Transformers 5.x training. Weight tensors stay unchanged by that metadata repair. Export verification compares vocabulary, chat-template token IDs across 12 language samples and all 86 evaluation prompts, and architecture/dtype fields in the actual serving environment.

The six lexical referral/refusal safety probes are a **limited smoke screen**. They can miss unsafe content or reject safe paraphrases; they do not replace multilingual toxicity, bias, hallucination or expert domain audits. Medical answers remain educational; government guidance needs official verification.

Do not claim a trained 7B model, 10+ language/code-mixed competence, full benchmark gains, completed independent safety review or production readiness without matching artifacts. The [architecture plan](architecture/00_MASTER_PLAN.md) remains the broader target; the report preserves missing stages and negative deltas.

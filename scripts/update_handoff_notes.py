"""Refresh current-machine notes from completed run artifacts, keeping pending work explicit."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    file = ROOT / path
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def note(
    file,
    title,
    implemented,
    manifest,
    workflow,
    config,
    dependencies,
    troubleshooting,
    verification,
):
    sections = [
        f"# {title}",
        "",
        "Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.",
        "",
    ]
    for heading, content in [
        ("Implemented and executed", implemented),
        ("File manifest", manifest),
        ("How it works", workflow),
        ("Configuration", config),
        ("Dependencies", dependencies),
        ("Troubleshooting", troubleshooting),
        ("Verification", verification),
    ]:
        sections.extend([f"## {heading}", "", content, ""])
    (ROOT / "memory" / file).write_text("\n".join(sections), encoding="utf-8")


note(
    "01_memory_environment.md",
    "Stage 01 - Environment",
    "Installed separate training (291 packages) and serving environments. Both passed uv pip check. verify_setup.py --full passed on the current GPU.",
    "scripts/setup_wsl.sh; scripts/install_wsl_dependencies.sh; scripts/wsl_run.sh; scripts/prepare_native_overlay.py; scripts/cache_package_metadata.py; scripts/install_cuda_compiler.py; scripts/link_cuda_compiler.sh; configs/requirements-*.txt and *.lock.txt.",
    "Large environments, model files and caches live at D:/local-llm-runtime. Python-heavy packages are mirrored into /home/krish/.cache/local-llm/native-packages/<environment>/site-packages and linked to the same compiled libraries. The literal site-packages path is required for dill/Hugging Face dataset fingerprinting. The wrapper activates a cache only after its .ready marker exists; regenerate after dependency changes.",
    "Ubuntu Python 3.12.3; RTX 4070 Laptop GPU, 8188 MiB VRAM. Train: AlignTune 0.1.28, Torch 2.11.0, Transformers 5.20.2, TRL 1.7.1, PEFT 0.21.2, bitsandbytes 0.50.2, CuratorKIT 0.1.22, AuditKIT 1.1.3, Gradio 6.30.0. Serve: SGLang 0.5.9, Torch 2.9.1, Transformers 4.57.1; verified NVIDIA CUDA compiler 12.8.93 and CCCL headers.",
    "CUDA driver must be visible in WSL. The two environments intentionally cannot be combined.",
    "Windows Python alias is unusable; host helper scripts ran with C:/Users/Krish/miniconda3/python.exe. Use scripts/wsl_run.sh for training. Do not install SGLang into the training environment. Native overlay symlinks require the D: environments to remain present. /tmp/local-llm is used for POSIX temporary-file semantics.",
    "Run wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/wsl_run.sh scripts/verify_setup.py --full; uv pip check --python <environment>/bin/python. Package presence alone is not workflow completion.",
)
data = read("results/reports/data_provenance.json")
note(
    "02_memory_data_collection.md",
    "Stage 02 - Data collection",
    json.dumps(data, ensure_ascii=False),
    "scripts/data/acquire_indic_subset.py; scripts/data/acquire_flores_eval.py; scripts/data/create_seed_data.py; results/reports/data_provenance.json; results/reports/flores_provenance.json.",
    "Freeze disjoint Aya train/validation/test splits by source-prompt hash. Cache successful per-language requests so retries do not erase acquired data. FLORES-200 is evaluation-only: first five aligned devtest examples per ten languages, never training data.",
    "216 Aya training, 36 validation and 36 test records across nine Indic languages. 50 FLORES-200 pilot records across ten languages. Eight handcrafted preference seeds have no independent human review.",
    "Source metadata and licence are recorded. Downloaded JSONL files are gitignored; regenerate with acquisition scripts.",
    "Aya was renamed to CohereLabs/aya_dataset. Its dataset-viewer filter sometimes returns HTTP 500; retries and cached source rows are supported. No available Odia Aya rows were found with the queried language codes. UI code-mixed/Odia support remains unproven.",
    "Inspect provenance JSONs and compare prompt_fingerprint sets before training. Do not sum cache, domain seed subsets, raw and derived exports as unique training examples.",
)
curation = {}
for stage in [
    "cleaned_sft",
    "qa_generation",
    "multiturn_generation",
    "preference_generation",
    "reviewed_preferences",
]:
    value = read(f"data/curated/{stage}/manifest.json")
    if value:
        curation[stage] = value["stage_counts"]
note(
    "03_memory_data_curation.md",
    "Stage 03 - CuratorKIT",
    json.dumps(curation, ensure_ascii=False),
    "configs/curatorkit/*.yaml; scripts/data/run_curation.py; scripts/data/merge_curated_data.py; scripts/training/format_sft_data.py; tests/test_toolkit_inputs.py; data/curated/<pipeline>/{manifest.json,checksums.txt,lexsi_provenance.json,rejected.jsonl}.",
    "Use the published PipelineConfig schema. Input schema checks run BEFORE generation. Generators use real SGLang. The grounding judge accepted two factual errors: Bengali triangle geometry and a Tamil dengue passage misread as a wedding. Source-bound agent review excludes both using configs/synthetic_review.json and checks raw export hashes. Seven DPO chosen answers use handwritten references paired with reviewed weaker generations. Merge only canonical accepted exports, deduplicate prompts and trim unanswered trailing user turns. Final training data: 220 SFT conversations and seven preferences; no independent expert review.",
    "SGLang base http://127.0.0.1:30000/v1, served model local-llm. Grounding threshold 0.6, judge failures rejected. Generated examples and same-model judgements are not independent factual/safety review.",
    "Current environments; frozen source records; SGLang must finish warm-up before LLM stages. Clean-only curation can run without an endpoint.",
    "High-level CuratorConfig YAML is not the CLI PipelineConfig schema. Pretrain reader consumes text, not a renamed output field. JSONLReader.read() returns (samples, rejected). Empty accepted exports cause run_curation.py to return failure. Do not weaken the grounding gate to hide rejected generations.",
    "Run curation and inspect every manifest stage count and rejection reason. Three toolkit integration tests exercise actual source readers/schema gates before generation. Merge/formatted outputs must be nonempty and disjoint from frozen evaluation prompts.",
)
for stage, filename, title in [
    ("sft", "04_memory_finetuning.md", "Stage 04 - AlignTune SFT"),
    ("dpo", "05_memory_dpo_alignment.md", "Stage 05 - AlignTune DPO"),
]:
    report = read(f"results/reports/local_{stage}_run.json")
    state = (
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"history", "result"}
            },
            ensure_ascii=False,
            default=str,
        )
        + ". Complete loss history is in the linked run report; final mean loss: "
        + str(
            next(
                (
                    entry["train_loss"]
                    for entry in reversed(report.get("history", []))
                    if "train_loss" in entry
                ),
                None,
            )
        )
        if report
        else "No completed local run artifact exists yet. Training is pending; do not infer success from a configured runner or historical Nix-host smoke notes."
    )
    note(
        filename,
        title,
        state,
        f"scripts/training/run_local_{stage}.py; scripts/training/merge_local_adapter.py; configs/aligntune/local_1_5b_sft.yaml; outputs/local_1_5b_{stage}; results/reports/local_{stage}_run.json.",
        "Use AlignTune's published trainer factory with its TRL backend and PEFT 4-bit LoRA. Save toolkit provenance, adapter/tokenizer, trainer state, loss history, global steps, elapsed time and peak allocated VRAM. Merge using AlignTune PEFTMerger.",
        "Explicit local Qwen2.5-1.5B-Instruct profile: sequence length 384, batch 1, accumulation 4, one epoch, LoRA rank 8/alpha 16, NF4 double quantization, BF16. SFT LR 1e-4; DPO LR 1e-5, beta 0.1. The 7B production config is separate and unexecuted here.",
        "SFT needs curated nonempty messages. DPO needs merged SFT weights and at least two distinct valid chosen/rejected pairs. Stop serving before either GPU training stage.",
        "The 7B profile is not suitable for this 8 GB GPU. Native package paths outside site-packages caused dill MonthDayNano serialization failure; the standard cache layout fixes it. Merging adds legacy RoPE/dtype/tokenizer metadata for Transformers 4.x serving without changing weight tensors. DPO toolkit result.total_steps is an unreliable zero; trainer.state.global_step records the actual two optimizer steps. Do not treat these small preference data as a safety guarantee.",
        f"Require results/reports/local_{stage}_run.json status completed, positive optimizer steps, finite losses and a reloadable adapter. Compare held-out outputs through SGLang after merging.",
    )
for filename, title, evidence, manifest in [
    (
        "06_memory_evaluation.md",
        "Stage 06 - Evaluation",
        "Completed evaluation artifacts: "
        + ", ".join(
            stage
            for stage in ["baseline", "sft", "dpo"]
            if read(f"results/reports/{stage}_evaluation.json")
        )
        + ". Each uses the same 36 Aya and 50 FLORES prompts plus six safety probes. Aya and FLORES pilots are not full benchmark results; lexical safety probes are not a toxicity/bias/hallucination audit.",
        "scripts/eval/local_evaluation.py; scripts/eval/run_evaluation.py; results/benchmarks/<stage>/; results/safety/<stage>/",
    ),
    (
        "07_memory_deployment.md",
        "Stage 07 - SGLang serving",
        "SGLang completed GPU warm-up and real generation for each model with a completed evaluation artifact. Export checks compare vocabulary, 12 language samples, all 86 evaluation prompts, chat-template token IDs, model architecture, RoPE and dtype. Inspect results/reports/{sft,dpo}_export_verification.json and the model path recorded by each evaluation.",
        "scripts/serve/local_server.py; scripts/serve/launch_server.py; results/reports/model_download.json; scripts/link_cuda_compiler.sh",
    ),
    (
        "08_memory_gui.md",
        "Stage 08 - GUI",
        "Gradio build, Unicode streaming, history preservation, empty input and unavailable-server handling are covered by tests. Live verification: "
        + json.dumps(
            read("results/reports/gui_verification.json") or {"status": "pending"},
            ensure_ascii=False,
        ),
        "src/gui/app.py; tests/test_gui.py",
    ),
    (
        "09_memory_benchmarking.md",
        "Stage 09 - Benchmark report",
        "The report generator writes only available measured artifacts. No missing training/evaluation stage is marked completed. Submission readiness requires representative benchmark gains and independent safety/native-speaker review.",
        "scripts/write_run_report.py; results/reports/FINAL_BENCHMARK_REPORT.md",
    ),
]:
    note(
        filename,
        title,
        evidence,
        manifest,
        "Use the current WSL wrapper and record artifacts from successful real executions. Preserve failures and negative deltas.",
        "Local endpoint 127.0.0.1:30000, GUI 127.0.0.1:7860. Base and candidate evaluation must use the same frozen prompts and generation settings.",
        "Depends on completed preceding stages and the current machine's artifacts.",
        "SGLang RoPE kernels require the full matching nvcc archive, runtime headers and CCCL; the pip nvcc package alone only provides ptxas. Original handoff's AMX error does not describe this rebuilt environment.",
        "Inspect actual reports, artifacts and live API/UI behavior before marking the stage finished.",
    )
handoff = [
    "# LocalLLM current handoff",
    "",
    "Updated 2026-10-25. This is the Windows/Ubuntu WSL2 continuation, replacing the previous Nix-host assumptions.",
    "",
    "## Verified",
    "",
    "- Both dependency environments are installed and compatible; GPU is RTX 4070 Laptop (8 GB).",
    "- Pinned base 1.5B model and source/evaluation data are downloaded with provenance. CuratorKIT cleaning, QA, multi-turn and preference generation completed; source-bound review excluded two factual errors. Training data contain 220 SFT conversations and seven reviewed preference pairs.",
    "- SGLang passed real GPU warm-up and generation after installing matching CUDA compiler/runtime/CCCL headers.",
    "- Data and Gradio tests pass; additional toolkit integration tests validate generation inputs.",
    "",
    "## Continue by evidence",
    "",
]
for stage in ["sft", "dpo"]:
    run = read(f"results/reports/local_{stage}_run.json")
    handoff.append(
        f"- {stage.upper()}: "
        + (
            f"completed, {run['global_steps']} optimizer steps"
            if run
            else "pending completed run artifact"
        )
    )
for stage in ["baseline", "sft", "dpo"]:
    evaluation = read(f"results/reports/{stage}_evaluation.json")
    handoff.append(
        f"- {stage} evaluation: "
        + (
            "completed; 86 quality predictions and six safety probes; mean ten-language FLORES pilot chrF++ "
            + f"{sum(v['chrF++'] for v in evaluation['suites']['flores']['per_language'].values()) / 10:.4f}"
            + f"; lexical safety score {evaluation['safety_keyword_screen']['referral_keyword_screen']:.4f}"
            if evaluation
            else "pending"
        )
    )
final = read("results/reports/final_verification.json")
gui = read("results/reports/gui_verification.json")
handoff.extend(
    [
        f"- Final technical verification: {final['status'] if final else 'pending'}; inspect results/reports/final_verification.json for command output and artifact-integrity checks.",
        f"- Live GUI/API: {gui['status'] if gui else 'pending'}, http://127.0.0.1:7860. Visual browser verification was blocked by computer-use access; API verification does not establish visual layout.",
        "- Checkpoint-specific AuditKIT caching includes weight SHA-256 hashes. Preserve actual negative benchmark deltas; shared API model names alone are insufficient cache identity.",
    ]
)
handoff.extend(
    [
        "",
        "## Commands",
        "",
        "Use `wsl -d Ubuntu -- bash /mnt/c/Users/Krish/local-llm/scripts/wsl_run.sh <script>` for train/tools. Prepend `env LOCALLLM_ENV=serve` after `--` for serving. See README.md for stage commands. Never run GPU training while SGLang occupies the GPU.",
        "",
        "## Remaining project targets",
        "",
        "The 7B target needs 16-24 GB VRAM. Full standard benchmarks, 10+ language/code-mixed training coverage, independent safety and native-speaker review, and representative benchmark gains remain separate from executing the small local pipeline. Do not claim these without new evidence.",
        "The PDF's PS4 outcome requires improvement, not merely executed training. Consult results/reports/FINAL_BENCHMARK_REPORT.md before making that claim; this pilot is too small for statistical superiority and any regression must remain visible. The PDF itself provides no numeric benchmark thresholds.",
        "",
    ]
)
(ROOT / "memory/10_remaining_work_handoff.md").write_text(
    "\n".join(handoff), encoding="utf-8"
)
print("Refreshed memory notes from current artifacts")

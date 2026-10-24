"""Build a candid handoff report from artifacts that actually exist."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


lines = [
    "# LocalLLM execution and benchmark report",
    "",
    "Updated 2026-10-25 (Asia/Kolkata). Values below come from local artifacts; missing stages remain pending.",
    "",
    "## Original PDF requirements",
    "",
    "The three-page LocalLLM PS.pdf requires one selected problem, SGLang as the core inference/serving engine, and CuratorKIT, AlignTune and AuditKIT across curation, fine-tuning and evaluation. PS4 (low-resource and multilingual public-facing language assistance) is selected. The PDF sets no numeric accuracy, BLEU, latency or throughput thresholds. The repository's 7B, 10+ language and benchmark-superiority goals are additional project targets.",
    "",
    "Requirement assessment: the selected PS4 and required toolchain have executable local evidence. CuratorKIT generation uses SGLang, AlignTune uses TRL for SFT/DPO, and AuditKIT evaluates the SGLang endpoint including six safety smoke probes. PS4 also asks for improvement in underserved languages: completion of training does not satisfy that outcome by itself. Pilot regressions are evidence against an improvement claim, and the small samples cannot establish representative public-facing quality. Independent multilingual safety evaluation remains incomplete. The project is therefore not established as submission-ready on quality grounds.",
    "",
    "## Current machine",
    "",
    "Windows with Ubuntu WSL2; RTX 4070 Laptop GPU with 8188 MiB VRAM. Python 3.12.3. Training and serving environments are isolated under D:/local-llm-runtime. This supersedes the earlier Nix/RTX 4050 handoff; earlier ignored model artifacts were not present in this checkout.",
    "",
    "## Data and model provenance",
    "",
]
for name in ["model_download", "data_provenance", "flores_provenance"]:
    report = read(f"results/reports/{name}.json")
    if report:
        lines.extend(
            [f"- `{name}.json`: " + json.dumps(report, ensure_ascii=False), ""]
        )
manifest = read("data/curated/cleaned_sft/manifest.json")
if manifest:
    lines.extend(
        [
            "## CuratorKIT cleaning",
            "",
            f"Executed CuratorKIT {manifest['tool_versions']['curatorkit']}: {json.dumps(manifest['stage_counts'])}. Manifest, checksums and provenance are in data/curated/cleaned_sft. Schema/cleaning/deduplication are not independent factual or safety review.",
            "",
        ]
    )
lines.extend(["## Training", ""])
lines.extend(
    [
        "CuratorKIT generated QA, multi-turn and preference candidates through SGLang. Its grounding judge accepted two factual errors (Bengali triangle geometry and Tamil dengue interpreted as a wedding). Source-bound agent review excludes both; raw generations and review hashes are preserved. Seven preferences anchor chosen answers to the repository's handwritten references. Final SFT data contain 220 conversations. This review is not independent expert annotation.",
        "",
    ]
)
for stage in ["sft", "dpo"]:
    report = read(f"results/reports/local_{stage}_run.json")
    if not report:
        lines.extend([f"- {stage.upper()}: no completed run artifact yet.", ""])
        continue
    values = {
        key: report[key]
        for key in [
            "status",
            "toolkit",
            "versions",
            "model",
            "model_revision",
            "dataset_count",
            "dataset_sha256",
            "duration_seconds",
            "peak_allocated_vram_bytes",
            "global_steps",
            "adapter_path",
            "limitations",
        ]
        if key in report
    }
    values["toolkit_provenance"] = f"outputs/local_1_5b_{stage}/lexsi_provenance.json"
    values["mean_training_loss"] = next(
        (
            entry["train_loss"]
            for entry in reversed(report.get("history", []))
            if "train_loss" in entry
        ),
        None,
    )
    lines.extend(
        [f"### {stage.upper()}", "", json.dumps(values, ensure_ascii=False), ""]
    )
evaluations = {}
lines.extend(
    [
        "## Evaluation",
        "",
        "Aya is a convenience holdout, not a standard benchmark. FLORES-200 uses a fixed first-five-devtest-sentences pilot per language, not the full benchmark. AuditKIT sentence BLEU/ROUGE-L/chrF use 0-1 scales. SacreBLEU spBLEU (FLORES-200 tokenizer) and chrF++ use 0-100 scales; exact signatures are recorded in each JSON. These tiny samples do not establish statistical superiority.",
        "",
    ]
)
for label in ["baseline", "sft", "dpo"]:
    report = read(f"results/reports/{label}_evaluation.json")
    if not report:
        lines.extend([f"- {label}: no completed evaluation artifact yet.", ""])
        continue
    evaluations[label] = report
    lines.extend([f"### {label}", ""])
    for suite, data in report["suites"].items():
        lines.extend(
            [
                f"- {suite}: n={data['count']}; AuditKIT metrics={json.dumps(data['auditkit_metrics'])}; performance={json.dumps(data['performance'])}."
            ]
        )
        for language, scores in data["per_language"].items():
            lines.append(f"  - {language}: {json.dumps(scores)}")
    lines.extend(
        [
            "",
            f"Safety smoke: {report['safety_count']} probes; keyword score={json.dumps(report['safety_keyword_screen'])}. {report['safety_limitations']}",
            "",
        ]
    )
if "baseline" in evaluations:
    for label in ["sft", "dpo"]:
        if label not in evaluations:
            continue
        lines.extend([f"### {label} minus baseline: FLORES chrF++ points", ""])
        for language, scores in (
            evaluations[label]["suites"]
            .get("flores", {})
            .get("per_language", {})
            .items()
        ):
            base = evaluations["baseline"]["suites"]["flores"]["per_language"][
                language
            ]["chrF++"]
            lines.append(f"- {language}: {scores['chrF++'] - base:+.4f} points.")
        lines.append("")
        candidate = evaluations[label]["suites"]["flores"]["per_language"]
        baseline = evaluations["baseline"]["suites"]["flores"]["per_language"]
        mean = sum(x["chrF++"] for x in candidate.values()) / len(candidate)
        base_mean = sum(x["chrF++"] for x in baseline.values()) / len(baseline)
        lines.extend(
            [
                f"Unweighted ten-language mean chrF++: baseline {base_mean:.4f}, {label} {mean:.4f}, difference {mean - base_mean:+.4f} points. This five-sentence-per-language pilot is descriptive only.",
                "",
            ]
        )
lines.extend(["## Final verification", ""])
for filename in [
    "sft_export_verification",
    "dpo_export_verification",
    "gui_verification_before_history_fix",
    "gui_verification",
    "final_verification",
]:
    evidence = read(f"results/reports/{filename}.json")
    lines.extend(
        [
            f"- {filename}: "
            + (json.dumps(evidence, ensure_ascii=False) if evidence else "pending"),
            "",
        ]
    )
gui = read("results/reports/gui_verification.json")
if gui and gui.get("arithmetic_answer_marker_present") is False:
    lines.extend([
        "The live GUI's functional API checks passed, but the model failed the simple Hindi 2+2 smoke question: it did not state four and produced an irrelevant question. This failure persisted after correcting Gradio 6 history text-block handling. The failed responses are preserved, and GUI functional success is not model-quality success.",
        "",
    ])
lines.extend(
    [
        "AuditKIT's API cache originally reused baseline predictions when different checkpoints shared local-llm. Evaluation now fingerprints actual safetensors contents and loaded model path, and a regression test checks separation even when weights change at the same path. The invalid reused SFT results were replaced by fresh requests; model-specific identity is recorded in the final JSON artifacts.",
        "",
    ]
)
lines.extend(
    [
        "## Remaining limits",
        "",
        "- The 7B production training target requires a larger GPU; only the explicitly named 1.5B local profile is appropriate here.",
        "- Full FLORES-200, MMLU and IndicXTREME evaluation and representative native-speaker public-service review are not replaced by these pilots.",
        "- Odia and meaningful code-mixed training coverage remain incomplete in Aya. A language appearing in a GUI dropdown does not demonstrate model competence.",
        "- Synthetic model judges and lexical safety probes are limited evidence. No clinical, governmental, toxicity, bias or hallucination safety certification is claimed.",
        "- Negative benchmark deltas must be preserved; an executed pipeline does not prove an improved model.",
        "",
        "## Evidence and reproduction",
        "",
        "See README.md for current WSL commands. Inspect results/reports/*.json, results/benchmarks/<stage>/, results/safety/<stage>/, outputs/local_1_5b_* and memory/10_remaining_work_handoff.md. A stage is complete only when its artifact and successful verification exist.",
        "",
    ]
)
(ROOT / "results/reports/FINAL_BENCHMARK_REPORT.md").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("Wrote results/reports/FINAL_BENCHMARK_REPORT.md from available artifacts")

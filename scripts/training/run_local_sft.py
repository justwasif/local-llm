"""Run real AlignTune/TRL QLoRA on the curated dataset and save measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_config, load_jsonl, prompt_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aligntune/local_1_5b_sft.yaml")
    args = parser.parse_args()
    import torch
    from aligntune.core.backend_factory import create_sft_trainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    data = PROJECT_ROOT / "data/curated/sft_messages_train.jsonl"
    rows = load_jsonl(data)
    if not rows:
        raise ValueError("No curated training data")
    test = load_jsonl(PROJECT_ROOT / "data/eval/aya_test.jsonl")
    prompts = {
        prompt_fingerprint(
            next(m["content"] for m in r["messages"] if m["role"] == "user")
        )
        for r in rows
    }
    if prompts & {prompt_fingerprint(r["instruction"]) for r in test}:
        raise ValueError("Training/test leakage detected")
    config = load_config(args.config)
    model_report = json.loads(
        (PROJECT_ROOT / "results/reports/model_download.json").read_text()
    )
    config["model_name"] = model_report["path"]
    output = PROJECT_ROOT / "outputs/local_1_5b_sft"
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = create_sft_trainer(
        dataset_name=str(data), output_dir=str(output), **config
    )
    trainer.train()
    trainer.tokenizer.save_pretrained(output)
    trainer.trainer.save_state()
    metrics = {
        "status": "completed",
        "toolkit": "aligntune",
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "aligntune",
                "torch",
                "transformers",
                "trl",
                "peft",
                "bitsandbytes",
            ]
        },
        "model": model_report["model"],
        "model_revision": model_report["revision"],
        "dataset_count": len(rows),
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "duration_seconds": time.time() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "global_steps": trainer.trainer.state.global_step,
        "history": trainer.trainer.state.log_history,
        "adapter_path": str(output),
        "limitations": "Small local proof-of-pipeline; held-out benchmark comparison required before claiming improvement.",
    }
    report = PROJECT_ROOT / "results/reports/local_sft_run.json"
    report.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

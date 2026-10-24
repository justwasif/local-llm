"""Run AlignTune DPO from the merged SFT model on accepted preference pairs."""

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_config, load_jsonl


def main():
    import torch
    from aligntune.core.backend_factory import create_rl_trainer

    data = PROJECT_ROOT / "data/preference/multilingual_pref.jsonl"
    rows = load_jsonl(data)
    if len(rows) < 2 or any(
        not all(r.get(k) for k in ["prompt", "chosen", "rejected"])
        or r["chosen"] == r["rejected"]
        for r in rows
    ):
        raise ValueError("DPO needs at least two valid, distinct preference pairs")
    base = PROJECT_ROOT / "outputs/local_1_5b_sft_merged"
    if not (base / "config.json").exists():
        raise FileNotFoundError("Merge the measured SFT adapter before DPO")
    config = load_config("configs/aligntune/local_1_5b_sft.yaml")
    for key in [
        "model_name",
        "completion_only_loss",
        "compute_rouge",
        "compute_bleu",
        "compute_perplexity",
    ]:
        config.pop(key, None)
    config.update(
        algorithm="dpo",
        learning_rate=1e-5,
        beta=0.1,
        warmup_ratio=0.0,
        max_prompt_length=192,
        max_length=384,
        max_completion_length=192,
        loss_type="sigmoid",
        bf16=True,
        dataset_num_proc=1,
    )
    output = PROJECT_ROOT / "outputs/local_1_5b_dpo"
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = create_rl_trainer(
        model_name=str(base), dataset_name=str(data), output_dir=str(output), **config
    )
    result = trainer.train()
    trainer.tokenizer.save_pretrained(output)
    trainer.trainer.save_state()
    report = {
        "status": "completed",
        "toolkit": "aligntune",
        "algorithm": "dpo",
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
        "model": str(base),
        "model_revision": json.loads(
            (PROJECT_ROOT / "results/reports/model_download.json").read_text()
        )["revision"],
        "dataset_count": len(rows),
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "duration_seconds": time.time() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "global_steps": trainer.trainer.state.global_step,
        "history": trainer.trainer.state.log_history,
        "adapter_path": str(output),
        "result": result,
        "limitations": "Seven source-anchored, agent-reviewed preference pairs; independent native-speaker review remains required. This does not establish safety. Use trainer.state global_steps rather than the toolkit result.total_steps fallback.",
    }
    (PROJECT_ROOT / "results/reports/local_dpo_run.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

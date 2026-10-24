"""Verify the completed local run and save concrete check output for handoff."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import load_jsonl, prompt_fingerprint


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    checks = {}

    def command(name, args):
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, check=False
        )
        checks[name] = {
            "exit_code": result.returncode,
            "output": (result.stdout + result.stderr).strip(),
        }
        if result.returncode:
            raise RuntimeError(f"{name} failed: {checks[name]}")

    command("pytest", [sys.executable, "-m", "pytest", "tests", "-q"])
    command(
        "compile", [sys.executable, "-m", "compileall", "-q", "scripts", "src", "tests"]
    )
    command(
        "python_lint",
        [sys.executable, "-m", "ruff", "check", "scripts", "src", "tests"],
    )
    command("setup", [sys.executable, "scripts/verify_setup.py", "--full"])
    command("jsonl_files", [sys.executable, "scripts/data/verify_data.py"])
    runtime = Path(os.environ.get("LOCALLLM_RUNTIME", "/mnt/d/local-llm-runtime"))
    for environment in ["train", "serve"]:
        command(
            f"dependencies_{environment}",
            [
                str(runtime / "tools/bin/uv"),
                "pip",
                "check",
                "--python",
                str(runtime / environment / "bin/python"),
            ],
        )
        frozen = subprocess.check_output(
            [
                str(runtime / "tools/bin/uv"),
                "pip",
                "freeze",
                "--python",
                str(runtime / environment / "bin/python"),
            ],
            text=True,
        )
        (ROOT / f"configs/requirements-{environment}.lock.txt").write_text(
            frozen, encoding="utf-8"
        )
    data_files = {
        "sft": "data/curated/sft_messages_train.jsonl",
        "dpo": "data/preference/multilingual_pref.jsonl",
    }
    train_prompts = set()
    for stage, path in data_files.items():
        rows = load_jsonl(ROOT / path)
        report = read(f"results/reports/local_{stage}_run.json")
        assert report["status"] == "completed" and report["global_steps"] > 0
        assert report["dataset_count"] == len(rows)
        assert all(
            math.isfinite(entry["loss"])
            for entry in report["history"]
            if "loss" in entry
        )
        digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if report.get("dataset_sha256"):
            assert digest == report["dataset_sha256"]
        checks[f"{stage}_training"] = {
            "global_steps": report["global_steps"],
            "dataset_count": len(rows),
            "validated_dataset_sha256": digest,
        }
        assert (
            read(f"results/reports/{stage}_export_verification.json")["status"]
            == "passed"
        )
        for row in rows:
            prompts = (
                [m["content"] for m in row["messages"] if m["role"] == "user"]
                if stage == "sft"
                else [row["prompt"]]
            )
            train_prompts.update(prompt_fingerprint(prompt) for prompt in prompts)
    validation = load_jsonl(ROOT / "data/eval/aya_validation.jsonl")
    assert len(validation) == 36
    assert not train_prompts & {
        prompt_fingerprint(row["instruction"]) for row in validation
    }
    all_identities = []
    for label in ["baseline", "sft", "dpo"]:
        summary = read(f"results/reports/{label}_evaluation.json")
        identity = summary["checkpoint_identity"]
        all_identities.append(identity["weights_sha256"])
        for suite, count in [("aya", 36), ("flores", 50)]:
            raw = read(f"results/benchmarks/{label}/{suite}/report.json")
            assert not raw["failed_count"] and not raw["errors"]
            assert len(raw["predictions"]) == count
            assert raw["metadata"]["checkpoint_identity"] == identity
            for prediction in raw["predictions"]:
                assert prediction["raw_output"].strip()
                assert prompt_fingerprint(prediction["prompt"]) not in train_prompts
        safety = read(f"results/safety/{label}/report.json")
        assert not safety["failed_count"] and not safety["errors"]
        assert len(safety["predictions"]) == 6
        assert safety["metadata"]["checkpoint_identity"] == identity
        checks[f"{label}_evaluation"] = {
            "quality_predictions": 86,
            "safety_predictions": 6,
            "checkpoint_identity": identity,
        }
    assert len({json.dumps(i, sort_keys=True) for i in all_identities}) == 3
    assert read("results/reports/gui_verification.json")["status"] == "passed"
    checks["data_leakage"] = (
        "No normalized SFT/DPO user prompt matches the 86 frozen quality-evaluation prompts or 36 validation prompts"
    )
    report = {
        "status": "passed",
        "checks": checks,
        "limitations": "Technical local-pipeline checks. Full benchmarks, independent safety/language review and 7B training are not completed. Browser visual verification was blocked by computer-use access.",
    }
    (ROOT / "results/reports/final_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

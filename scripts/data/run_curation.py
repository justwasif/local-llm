"""Run CuratorKIT pipelines after confirming the local SGLang endpoint is ready."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_config, load_jsonl, setup_logging

LOGGER = setup_logging("curation", "curation.log")


def server_ready(url: str) -> bool:
    try:
        return requests.get(url, timeout=5).ok
    except requests.RequestException:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["clean", "qa", "multiturn", "preference", "reviewed"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config()
    url = (
        os.getenv(
            "SGLANG_BASE_URL", f"http://127.0.0.1:{config['sglang']['port']}/v1"
        ).rstrip("/")
        + "/models"
    )
    if not args.dry_run and args.only not in {"clean", "reviewed"} and not server_ready(url):
        LOGGER.error(
            "SGLang is unavailable at %s. Start scripts/serve/launch_server.py first.",
            url,
        )
        return 2
    failures = []
    pipelines = {
        "clean": ("clean_sft_data.yaml", "cleaned_sft"),
        "qa": ("qa_generation.yaml", "qa_generation"),
        "multiturn": ("multiturn_generation.yaml", "multiturn_generation"),
        "preference": ("preference_generation.yaml", "preference_generation"),
    }
    if args.only == "reviewed":
        pipelines = {"reviewed": ("reviewed_preferences.yaml", "reviewed_preferences")}
    for key, (name, output) in pipelines.items():
        if args.only and key != args.only:
            continue
        path = PROJECT_ROOT / "configs/curatorkit" / name
        command = [
            "curatorkit",
            "run",
            str(path),
            "--output-dir",
            str(PROJECT_ROOT / "data/curated" / output),
        ]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            failures.append(name)
        elif not args.dry_run:
            export_name = "dpo.jsonl" if key in {"preference", "reviewed"} else "sft_messages.jsonl"
            export = PROJECT_ROOT / "data/curated" / output / export_name
            if not export.exists() or not load_jsonl(export):
                LOGGER.error("Pipeline %s produced no accepted records", name)
                failures.append(name)
    if failures:
        LOGGER.error("Failed pipelines: %s", ", ".join(failures))
        return 1
    LOGGER.info("All CuratorKIT pipelines completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

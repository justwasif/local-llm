"""Run CuratorKIT pipelines after confirming the local SGLang endpoint is ready."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_config, setup_logging

LOGGER = setup_logging("curation", "curation.log")

def server_ready(url: str) -> bool:
    try: return requests.get(url, timeout=5).ok
    except requests.RequestException: return False

def main() -> int:
    config = load_config(); url = f"http://localhost:{config['sglang']['port']}/v1/models"
    if not server_ready(url):
        LOGGER.error("SGLang is unavailable at %s. Start scripts/serve/launch_server.py first.", url)
        return 2
    failures = []
    for name in ("clean_sft_data.yaml", "qa_generation.yaml", "multiturn_generation.yaml", "preference_generation.yaml"):
        path = PROJECT_ROOT / "configs/curatorkit" / name
        result = subprocess.run(["curatorkit", "run", str(path)], cwd=PROJECT_ROOT)
        if result.returncode: failures.append(name)
    if failures:
        LOGGER.error("Failed pipelines: %s", ", ".join(failures)); return 1
    LOGGER.info("All CuratorKIT pipelines completed."); return 0
if __name__ == "__main__": raise SystemExit(main())

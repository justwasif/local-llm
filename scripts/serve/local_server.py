"""Bounded SGLang launch for the 8 GB local GPU; keep only one GPU stage active."""

import argparse
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--model-path")
parser.add_argument("--log-file", default="logs/sglang.log")
args = parser.parse_args()
model = (
    args.model_path
    or json.loads((root / "results/reports/model_download.json").read_text())["path"]
)
runtime = Path(os.environ.get("LOCALLLM_RUNTIME", "/mnt/d/local-llm-runtime"))
command = [
    str(runtime / "serve/bin/python"),
    "-m",
    "sglang.launch_server",
    "--model-path",
    model,
    "--served-model-name",
    "local-llm",
    "--host",
    "127.0.0.1",
    "--port",
    "30000",
    "--context-length",
    "2048",
    "--mem-fraction-static",
    "0.65",
    "--max-running-requests",
    "2",
    "--chunked-prefill-size",
    "512",
    "--attention-backend",
    "torch_native",
    "--disable-cuda-graph",
    "--disable-custom-all-reduce",
    "--random-seed",
    "42",
]
print("Launching", command, flush=True)
log_path = root / args.log_file
log_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Full server output: {log_path}", flush=True)
with log_path.open("a", encoding="utf-8") as log:
    log.write("\nCOMMAND: " + repr(command) + "\n")
    log.flush()
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        log.write(line)
        log.flush()
        if any(
            marker in line
            for marker in [
                "ready to roll",
                "Traceback",
                "Scheduler hit",
                "ERROR",
                "Uvicorn running",
                "Load weight end",
            ]
        ):
            print(line, end="", flush=True)
    raise SystemExit(process.wait())

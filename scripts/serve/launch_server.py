"""Launch an OpenAI-compatible SGLang server using the final model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--quantization", default=None)
    args = parser.parse_args()
    cfg = load_config()
    model = args.model_path or str(PROJECT_ROOT / cfg["paths"]["output_merged"])
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        cfg["sglang"]["host"],
        "--port",
        str(cfg["sglang"]["port"]),
        "--mem-fraction-static",
        str(cfg["sglang"]["mem_fraction_static"]),
    ]
    if args.quantization:
        command.extend(["--quantization", args.quantization])
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

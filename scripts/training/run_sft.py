"""Launch SFT through AlignTune; fail early if formatted data is absent."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_true", help="Use the measured 1.5B/8 GB profile"
    )
    args = parser.parse_args()
    if args.local:
        from run_local_sft import main as local_main

        sys.argv = [sys.argv[0]]
        local_main()
        return 0
    from utils import get_gpu_info

    gpu = get_gpu_info()
    if not gpu.get("available") or gpu.get("total_memory_gb", 0) < 16:
        print(
            "The 7B production profile requires at least 16 GB VRAM. Use --local for 1.5B.",
            file=sys.stderr,
        )
        return 2
    data = PROJECT_ROOT / "data/curated/sft_messages_train.jsonl"
    if not data.exists() or not data.stat().st_size:
        print(
            "Format curated data first: python scripts/training/format_sft_data.py",
            file=sys.stderr,
        )
        return 2
    if not shutil.which("aligntune"):
        print(
            "AlignTune is not installed. Run pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    return subprocess.run(
        ["aligntune", "train", "--config", "configs/aligntune/sft_config.yaml"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

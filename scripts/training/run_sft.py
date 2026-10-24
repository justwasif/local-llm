"""Launch SFT through AlignTune; fail early if formatted data is absent."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT
def main() -> int:
    data = PROJECT_ROOT / "data/curated/sft_messages_train.jsonl"
    if not data.exists() or not data.stat().st_size: print("Format curated data first: python scripts/training/format_sft_data.py", file=sys.stderr); return 2
    if not shutil.which("aligntune"): print("AlignTune is not installed. Run pip install -r requirements.txt", file=sys.stderr); return 2
    return subprocess.run(["aligntune", "train", "--config", "configs/aligntune/sft_config.yaml"], cwd=PROJECT_ROOT).returncode
if __name__ == "__main__": raise SystemExit(main())

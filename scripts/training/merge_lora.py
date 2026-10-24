"""Merge the final DPO adapter into its SFT base using AlignTune."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT
def main() -> int:
    base, adapter, output = PROJECT_ROOT / "outputs/sft_model/final", PROJECT_ROOT / "outputs/dpo_model/checkpoint-final", PROJECT_ROOT / "outputs/merged_model"
    if not base.exists() or not adapter.exists(): print("Expected SFT base and final DPO adapter are missing.", file=sys.stderr); return 2
    if not shutil.which("aligntune"): print("AlignTune is not installed.", file=sys.stderr); return 2
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["aligntune", "merge", "--base-model", str(base), "--adapter", str(adapter), "--output", str(output)], cwd=PROJECT_ROOT).returncode
if __name__ == "__main__": raise SystemExit(main())

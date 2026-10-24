"""Launch DPO alignment from the SFT adapter."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT
def main() -> int:
    data = PROJECT_ROOT / "data/preference/multilingual_pref.jsonl"; sft = PROJECT_ROOT / "outputs/sft_model/final"
    if not data.exists() or not sft.exists(): print("DPO requires data/preference/multilingual_pref.jsonl and outputs/sft_model/final", file=sys.stderr); return 2
    if not shutil.which("aligntune"): print("AlignTune is not installed.", file=sys.stderr); return 2
    return subprocess.run(["aligntune", "train", "--config", "configs/aligntune/dpo_config.yaml", "--method", "dpo"], cwd=PROJECT_ROOT).returncode
if __name__ == "__main__": raise SystemExit(main())

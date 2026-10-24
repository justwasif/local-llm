"""Merge a measured local adapter with AlignTune's supported PEFT merger."""

import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))
from serve.ensure_legacy_config import ensure_compatible

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["sft", "dpo"], default="sft")
args = parser.parse_args()
from aligntune.core.merge.peft_merger import PEFTMerger

base = (
    json.loads((root / "results/reports/model_download.json").read_text())["path"]
    if args.stage == "sft"
    else str(root / "outputs/local_1_5b_sft_merged")
)
adapter = root / f"outputs/local_1_5b_{args.stage}"
if not (adapter / "adapter_config.json").is_file():
    raise FileNotFoundError(f"No trained adapter at {adapter}")
output = root / f"outputs/local_1_5b_{args.stage}_merged"
PEFTMerger().merge_lora(
    base_model=base,
    adapter_path=str(adapter),
    output_path=str(output),
    torch_dtype="bfloat16",
)
ensure_compatible(output)
print(f"Merged model: {output}")

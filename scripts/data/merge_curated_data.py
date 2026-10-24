"""Normalize and merge CuratorKIT outputs into SFT and DPO JSONL files."""
from __future__ import annotations
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl, save_jsonl, setup_logging
LOGGER = setup_logging("merge_data", "merge_data.log")

def records_under(root: Path):
    for file in root.rglob("*.jsonl") if root.exists() else []:
        if "rejected" not in file.name:
            yield from load_jsonl(file)

def valid_sft(row: dict) -> bool:
    return bool(row.get("messages") or (row.get("instruction") and row.get("output")) or row.get("conversations"))

def valid_dpo(row: dict) -> bool:
    return bool(row.get("prompt") and row.get("chosen") and row.get("rejected"))

def main() -> None:
    curated = PROJECT_ROOT / "data/curated"
    sft = [row for folder in ("cleaned_sft", "qa_generation", "multiturn_generation") for row in records_under(curated / folder) if valid_sft(row)]
    dpo = [row for row in records_under(curated / "preference_generation") if valid_dpo(row)]
    random.Random(42).shuffle(sft); random.Random(42).shuffle(dpo)
    save_jsonl(sft, curated / "final_sft_train.jsonl")
    save_jsonl(dpo, PROJECT_ROOT / "data/preference/multilingual_pref.jsonl")
    LOGGER.info("Merged %d SFT and %d DPO records.", len(sft), len(dpo))
if __name__ == "__main__": main()

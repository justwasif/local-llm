"""Normalize and merge CuratorKIT outputs into SFT and DPO JSONL files."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    PROJECT_ROOT,
    load_jsonl,
    prompt_fingerprint,
    save_jsonl,
    setup_logging,
)

LOGGER = setup_logging("merge_data", "merge_data.log")


def records_under(root: Path):
    filename = (
        "dpo.jsonl"
        if root.name in {"preference_generation", "reviewed_preferences"}
        else "sft_messages.jsonl"
    )
    file = root / filename
    if file.is_file():
        yield from load_jsonl(file)


def valid_sft(row: dict) -> bool:
    return bool(
        row.get("messages")
        or (row.get("instruction") and row.get("output"))
        or row.get("conversations")
    )


def valid_dpo(row: dict) -> bool:
    return bool(
        row.get("prompt")
        and row.get("chosen")
        and row.get("rejected")
        and row["chosen"] != row["rejected"]
    )


def main() -> None:
    curated = PROJECT_ROOT / "data/curated"
    review_file = PROJECT_ROOT / "configs/synthetic_review.json"
    excluded = {}
    if review_file.exists():
        review = json.loads(review_file.read_text(encoding="utf-8"))
        for path, expected in review["reviewed_export_sha256"].items():
            if (
                hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Synthetic exports changed; re-review before merging")
        excluded = review["excluded_prompt_fingerprints"]
    candidates = [
        row
        for folder in ("cleaned_sft", "qa_generation", "multiturn_generation")
        for row in records_under(curated / folder)
        if valid_sft(row)
    ]
    raw_path = PROJECT_ROOT / "data/raw/sft/indic_aya_train.jsonl"
    metadata = (
        {prompt_fingerprint(r["instruction"]): r for r in load_jsonl(raw_path)}
        if raw_path.exists()
        else {}
    )
    sft, seen = [], set()
    for row in candidates:
        prompt = next(
            (m["content"] for m in row.get("messages", []) if m.get("role") == "user"),
            row.get("instruction", ""),
        )
        key = prompt_fingerprint(prompt)
        if key in excluded:
            LOGGER.warning("Excluded reviewed synthetic failure: %s", excluded[key])
            continue
        if key not in seen:
            original = metadata.get(key, {})
            row = dict(row) | {
                k: original[k]
                for k in ["language", "domain", "source", "license", "source_row"]
                if k in original
            }
            seen.add(key)
            sft.append(row)
    dpo = [
        row for row in records_under(curated / "reviewed_preferences") if valid_dpo(row)
    ]
    if not sft:
        raise ValueError(
            "No accepted SFT records; run CuratorKIT successfully before merging"
        )
    random.Random(42).shuffle(sft)
    random.Random(42).shuffle(dpo)
    save_jsonl(sft, curated / "final_sft_train.jsonl")
    save_jsonl(dpo, PROJECT_ROOT / "data/preference/multilingual_pref.jsonl")
    LOGGER.info("Merged %d SFT and %d DPO records.", len(sft), len(dpo))


if __name__ == "__main__":
    main()

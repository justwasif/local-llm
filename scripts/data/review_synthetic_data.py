"""Apply the recorded agent review, then anchor DPO chosen answers to repository references.

Raw model outputs remain unchanged. This is a small, source-anchored preference
construction, not independent expert annotation or a model-quality result.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl, prompt_fingerprint, save_jsonl


def main():
    review = json.loads(
        (PROJECT_ROOT / "configs/synthetic_review.json").read_text(encoding="utf-8")
    )
    for path, expected_hash in review["reviewed_export_sha256"].items():
        if (
            hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise ValueError(
                "Synthetic outputs changed; inspect them and update the recorded review before training"
            )
    seeds = {
        r["instruction"]: r
        for r in load_jsonl(PROJECT_ROOT / "data/raw/seeds/preference_seeds.jsonl")
    }
    generated = load_jsonl(
        PROJECT_ROOT / "data/curated/preference_generation/dpo.jsonl"
    )
    pairs = []
    for row in generated:
        seed = seeds[row["prompt"]]
        # The Telugu/Punjabi and Hinglish generated 'chosen' responses were weak.
        # A known repository reference is the chosen answer; generated rejected
        # text is retained only when clearly weaker under the recorded review.
        if row["prompt"] not in review["preference_prompts_with_weaker_rejected"]:
            continue
        chosen, rejected = seed["output"], row["rejected"]
        if prompt_fingerprint(chosen) == prompt_fingerprint(rejected):
            continue
        pairs.append(
            {
                "prompt": row["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "language": seed["language"],
                "chosen_source": "repository_handcrafted_reference",
                "rejected_source": "CuratorKIT/SGLang generation",
                "review_status": "agent-reviewed; not independently expert-reviewed",
            }
        )
    if len(pairs) < 2:
        raise ValueError("Insufficient source-anchored preference pairs")
    save_jsonl(pairs, PROJECT_ROOT / "data/preference/source_anchored_pairs.jsonl")
    print(
        f"Prepared {len(pairs)} source-anchored preferences; raw generations preserved"
    )


if __name__ == "__main__":
    main()

"""Record the specific agent review of this run; assert reviewed failure fragments."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl, prompt_fingerprint

paths = {
    stage: PROJECT_ROOT / f"data/curated/{stage}/sft_messages.jsonl"
    for stage in ["qa_generation", "multiturn_generation"]
}
qa, multi = (
    load_jsonl(paths["qa_generation"]),
    load_jsonl(paths["multiturn_generation"]),
)
assert "বাহুর দৈর্ঘ্য একই" in qa[0]["messages"][1]["content"], (
    "Generation differs; review the new outputs"
)
assert "marriage of the king" in multi[2]["messages"][1]["content"], (
    "Generation differs; review the new outputs"
)
paths["preference_generation"] = (
    PROJECT_ROOT / "data/curated/preference_generation/dpo.jsonl"
)
preferences = load_jsonl(paths["preference_generation"])
review = {
    "reviewer": "coding agent; not independent native-speaker/domain-expert review",
    "date": "2026-10-25",
    "reviewed_export_sha256": {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths.values()
    },
    "excluded_prompt_fingerprints": {
        prompt_fingerprint(
            qa[0]["messages"][0]["content"]
        ): "Incorrectly says all sides of a right triangle have the same length",
        prompt_fingerprint(
            multi[2]["messages"][0]["content"]
        ): "Misreads Tamil dengue source as a king's marriage; factual hallucination",
    },
    "conversation_policy": "Retain only complete alternating user/assistant exchanges; drop trailing unanswered user turns",
    "preference_prompts_with_weaker_rejected": [
        row["prompt"] for index, row in enumerate(preferences) if index != 1
    ],
    "preference_policy": "Use repository reference as chosen; retain reviewed weaker model-generated rejected text. Exclude the Bengali pair because both generated responses already give the same adequate content as the reference.",
    "limitations": "Small source-anchored preference construction; not independent expert annotation or proof of safety/quality.",
}
target = PROJECT_ROOT / "configs/synthetic_review.json"
target.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Recorded review in {target}")

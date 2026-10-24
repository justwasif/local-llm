"""Normalize merged records into Qwen-compatible messages training format."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl, save_jsonl

SYSTEM_PROMPT = """You are a helpful, accurate multilingual assistant for Indian citizens. You support education, healthcare, agriculture, and public services. Reply in the user's language. For medical, legal, or government matters, state limits and direct users to qualified or official sources when appropriate."""


def normalize(row: dict) -> dict | None:
    messages = (
        [dict(message) for message in row["messages"]] if row.get("messages") else None
    )
    if not messages and row.get("conversations"):
        messages = [
            {
                "role": "user"
                if turn.get("from") in ("human", "user")
                else "assistant",
                "content": turn.get("value", turn.get("content", "")),
            }
            for turn in row["conversations"]
        ]
    if not messages and row.get("instruction") and row.get("output"):
        prompt = row["instruction"] + (
            ("\n\n" + row["input"]) if row.get("input") else ""
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["output"]},
        ]
    if not messages or not any(
        m.get("role") == "assistant" and m.get("content") for m in messages
    ):
        return None
    # Preserve complete exchanges, never train an unanswered trailing user turn.
    while messages and messages[-1].get("role") == "user":
        messages.pop()
    turns = (
        messages[1:] if messages and messages[0].get("role") == "system" else messages
    )
    if not turns or len(turns) % 2:
        return None
    for index, message in enumerate(turns):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if (
            message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            return None
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {
        "messages": messages,
        "language": row.get("language", row.get("language_code", "unknown")),
        "domain": row.get("domain", "general"),
        "source": row.get("source", "unknown"),
    }


def main() -> None:
    source = PROJECT_ROOT / "data/curated/final_sft_train.jsonl"
    target = PROJECT_ROOT / "data/curated/sft_messages_train.jsonl"
    rows = [formatted for row in load_jsonl(source) if (formatted := normalize(row))]
    if not rows:
        raise ValueError(
            "No valid assistant conversations; refusing to write empty training data"
        )
    save_jsonl(rows, target)
    print(f"Formatted {len(rows)} samples -> {target}")


if __name__ == "__main__":
    main()

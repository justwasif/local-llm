"""Normalize merged records into Qwen-compatible messages training format."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl, save_jsonl

SYSTEM_PROMPT = """You are a helpful, accurate multilingual assistant for Indian citizens. You support education, healthcare, agriculture, and public services. Reply in the user's language. For medical, legal, or government matters, state limits and direct users to qualified or official sources when appropriate."""

def normalize(row: dict) -> dict | None:
    messages = row.get("messages")
    if not messages and row.get("conversations"):
        messages = [{"role": "user" if turn.get("from") in ("human", "user") else "assistant", "content": turn.get("value", turn.get("content", ""))} for turn in row["conversations"]]
    if not messages and row.get("instruction") and row.get("output"):
        prompt = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}]
    if not messages or not any(m.get("role") == "assistant" and m.get("content") for m in messages): return None
    if messages[0].get("role") != "system": messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"messages": messages, "language": row.get("language", row.get("language_code", "unknown")), "domain": row.get("domain", "general"), "source": row.get("source", "unknown")}

def main() -> None:
    source = PROJECT_ROOT / "data/curated/final_sft_train.jsonl"; target = PROJECT_ROOT / "data/curated/sft_messages_train.jsonl"
    rows = [formatted for row in load_jsonl(source) if (formatted := normalize(row))]
    save_jsonl(rows, target); print(f"Formatted {len(rows)} samples -> {target}")
if __name__ == "__main__": main()

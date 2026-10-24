import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


fmt = module("format_sft", "scripts/training/format_sft_data.py")
merge = module("merge_data", "scripts/data/merge_curated_data.py")


def test_normalize_does_not_modify_original_messages():
    row = {
        "messages": [
            {"role": "user", "content": "नमस्ते"},
            {"role": "assistant", "content": "नमस्कार"},
        ]
    }
    result = fmt.normalize(row)
    assert len(row["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert fmt.normalize(row) == result


def test_reject_missing_assistant():
    assert fmt.normalize({"messages": [{"role": "user", "content": "hello"}]}) is None


def test_merge_uses_only_canonical_export(tmp_path):
    folder = tmp_path / "cleaned_sft"
    folder.mkdir()
    accepted = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }
    for name in [
        "sft_messages.jsonl",
        "sft_alpaca.jsonl",
        "rejected.jsonl",
        "manifest.jsonl",
    ]:
        (folder / name).write_text(json.dumps(accepted) + "\n", encoding="utf-8")
    assert list(merge.records_under(folder)) == [accepted]


def test_empty_merge_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(merge, "PROJECT_ROOT", tmp_path)
    import pytest

    with pytest.raises(ValueError, match="No accepted SFT"):
        merge.main()


def test_jsonl_reports_bad_line(tmp_path):
    import pytest
    from utils import load_jsonl

    file = tmp_path / "bad.jsonl"
    file.write_text("{}\nBAD\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        load_jsonl(file)


def test_prompt_identity_normalizes_unicode_and_whitespace():
    from utils import prompt_fingerprint

    assert prompt_fingerprint("  Hello\nCAFÉ ") == prompt_fingerprint(
        "hello cafe\u0301"
    )


def test_identical_preference_pair_is_rejected():
    assert not merge.valid_dpo({"prompt": "Q", "chosen": "A", "rejected": "A"})


def test_unanswered_trailing_turn_is_removed_without_mutating_source():
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Unanswered"},
        ]
    }
    result = fmt.normalize(row)
    assert result["messages"][-1] == {"role": "assistant", "content": "A"}
    assert len(row["messages"]) == 3


def test_invalid_role_sequence_is_rejected():
    assert (
        fmt.normalize(
            {
                "messages": [
                    {"role": "assistant", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            }
        )
        is None
    )

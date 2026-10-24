"""Acquire a bounded, language-filtered Aya subset with frozen disjoint splits."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, save_jsonl

LANGUAGES = {
    "hin": "hi",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "mar": "mr",
    "guj": "gu",
    "kan": "kn",
    "mal": "ml",
    "pan": "pa",
    "ori": "or",
}
DATASET = "CohereLabs/aya_dataset"


def acquire(code):
    cached = PROJECT_ROOT / f"data/raw/aya_cache/{code}.jsonl"
    if cached.exists():
        return [
            json.loads(line)
            for line in cached.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "where": f"\"language_code\"='{code}'",
        "length": 100,
    }
    for attempt in range(3):
        response = requests.get(
            "https://datasets-server.huggingface.co/filter", params=params, timeout=120
        )
        if response.ok:
            break
        if attempt == 2:
            response.raise_for_status()
        time.sleep(2)
    rows = []
    seen = set()
    for item in response.json()["rows"]:
        raw = item["row"]
        prompt, answer = raw["inputs"].strip(), raw["targets"].strip()
        if not prompt or not answer or len(prompt) + len(answer) > 1800:
            continue
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": key,
                "instruction": prompt,
                "output": answer,
                "input": "",
                "language": LANGUAGES[code],
                "domain": "general",
                "source": DATASET,
                "source_row": item["row_idx"],
                "license": "apache-2.0",
            }
        )
        if len(rows) >= 32:
            break
    if len(rows) < 12:
        raise ValueError(f"Insufficient usable {code} rows: {len(rows)}")
    save_jsonl(rows, cached)
    return rows


def main():
    # Preserve already acquired source rows across transient endpoint failures.
    prior = []
    for name in [
        "data/raw/sft/indic_aya_train.jsonl",
        "data/eval/aya_validation.jsonl",
        "data/eval/aya_test.jsonl",
    ]:
        path = PROJECT_ROOT / name
        if path.exists():
            prior.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    for code, language in LANGUAGES.items():
        subset = [r for r in prior if r["language"] == language]
        if subset:
            save_jsonl(subset, PROJECT_ROOT / f"data/raw/aya_cache/{code}.jsonl")
    metadata = requests.get(
        f"https://huggingface.co/api/datasets/{DATASET}", timeout=30
    )
    metadata.raise_for_status()
    info = metadata.json()
    if info["cardData"]["license"] != "apache-2.0":
        raise ValueError("Dataset licence changed; review before training")
    train, validation, test, failures = [], [], [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(acquire, code): code for code in LANGUAGES}
        for future in concurrent.futures.as_completed(futures):
            code = futures[future]
            try:
                rows = future.result()
                rows.sort(key=lambda row: row["id"])
                test.extend(rows[:4])
                validation.extend(rows[4:8])
                train.extend(rows[8:])
                print(
                    f"{code}: {len(rows) - 8} train / 4 validation / 4 test", flush=True
                )
            except (requests.RequestException, ValueError, KeyError) as error:
                failures[code] = str(error)
                print(f"{code}: {error}", flush=True)
    if not train:
        raise RuntimeError(f"All language acquisitions failed: {failures}")
    for rows, path in [
        (train, "data/raw/sft/indic_aya_train.jsonl"),
        (validation, "data/eval/aya_validation.jsonl"),
        (test, "data/eval/aya_test.jsonl"),
    ]:
        save_jsonl(sorted(rows, key=lambda row: row["id"]), PROJECT_ROOT / path)
    report = {
        "dataset": DATASET,
        "revision": info["sha"],
        "license": info["cardData"]["license"],
        "acquisition": "HF dataset viewer filtered by language_code; at most 100 rows examined per language",
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "train_languages": dict(Counter(r["language"] for r in train)),
        "failures": failures,
        "limitations": "Small convenience sample, not a representative benchmark. General instructions, not verified public-service advice. No Hinglish/Banglish coverage from Aya.",
    }
    output = PROJECT_ROOT / "results/reports/data_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

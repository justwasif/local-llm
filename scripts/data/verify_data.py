"""Report JSONL datasets, sample count, and language coverage."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, load_jsonl


def main() -> int:
    total, languages = 0, Counter()
    for directory in ("data/raw", "data/curated", "data/preference", "data/eval"):
        root = PROJECT_ROOT / directory
        print(f"\n{directory}")
        for file in sorted(root.rglob("*.jsonl")):
            try:
                rows = load_jsonl(file)
                total += len(rows)
                languages.update(
                    row.get("language", row.get("language_code", "unknown"))
                    for row in rows
                )
                print(f"  {file.relative_to(PROJECT_ROOT)}: {len(rows)}")
            except ValueError as error:
                print(f"  INVALID {file}: {error}")
                return 1
    print(
        f"\nSerialized rows across sources, caches and exports (not unique training examples): {total}\nRow language metadata across those files: {dict(sorted(languages.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

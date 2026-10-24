"""Download configured public datasets with per-source failure isolation.

The script intentionally does not run at install time: datasets are large and may
require Hugging Face authentication. Use --limit for a smoke-sized acquisition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from create_seed_data import main as create_seed_data
from utils import PROJECT_ROOT, save_jsonl, setup_logging

LOGGER = setup_logging("data_download", "data_download.log")


def export_dataset(dataset_id: str, split: str, output: Path, limit: int) -> int:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    rows = []
    for index, row in enumerate(dataset):
        if limit and index >= limit:
            break
        rows.append(dict(row) | {"source": dataset_id})
    return save_jsonl(rows, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Rows per source; 0 uses all streamed rows (not recommended).",
    )
    parser.add_argument("--include-preferences", action="store_true")
    args = parser.parse_args()
    sources = [
        ("iamshnoo/alpaca-cleaned-hindi", "train", "alpaca_hindi.jsonl"),
        ("CohereForAI/aya_dataset", "train", "aya_dataset.jsonl"),
    ]
    target = PROJECT_ROOT / "data/raw/sft"
    successful = 0
    for dataset_id, split, filename in sources:
        try:
            count = export_dataset(dataset_id, split, target / filename, args.limit)
            LOGGER.info("Saved %d rows from %s", count, dataset_id)
            successful += 1
        except Exception as error:  # noqa: BLE001 - optional legacy source boundary; log failure and retain seeds
            LOGGER.warning("Could not download %s: %s", dataset_id, error)
    if args.include_preferences:
        try:
            count = export_dataset(
                "Anthropic/hh-rlhf",
                "train",
                PROJECT_ROOT / "data/raw/preference/hh_rlhf.jsonl",
                args.limit,
            )
            LOGGER.info("Saved %d preference pairs", count)
        except Exception as error:  # noqa: BLE001 - optional legacy preference source; log failure
            LOGGER.warning("Could not download preference data: %s", error)
    create_seed_data()
    LOGGER.info(
        "Finished with %d downloaded source(s); seed data is always available.",
        successful,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

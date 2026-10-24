# Stage 02 - Data collection

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

{"dataset": "CohereLabs/aya_dataset", "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2", "license": "apache-2.0", "acquisition": "HF dataset viewer filtered by language_code; at most 100 rows examined per language", "counts": {"train": 216, "validation": 36, "test": 36}, "train_languages": {"hi": 24, "te": 24, "bn": 24, "kn": 24, "mr": 24, "ml": 24, "pa": 24, "gu": 24, "ta": 24}, "failures": {"ori": "Insufficient usable ori rows: 0"}, "limitations": "Small convenience sample, not a representative benchmark. General instructions, not verified public-service advice. No Hinglish/Banglish coverage from Aya."}

## File manifest

scripts/data/acquire_indic_subset.py; scripts/data/acquire_flores_eval.py; scripts/data/create_seed_data.py; results/reports/data_provenance.json; results/reports/flores_provenance.json.

## How it works

Freeze disjoint Aya train/validation/test splits by source-prompt hash. Cache successful per-language requests so retries do not erase acquired data. FLORES-200 is evaluation-only: first five aligned devtest examples per ten languages, never training data.

## Configuration

216 Aya training, 36 validation and 36 test records across nine Indic languages. 50 FLORES-200 pilot records across ten languages. Eight handcrafted preference seeds have no independent human review.

## Dependencies

Source metadata and licence are recorded. Downloaded JSONL files are gitignored; regenerate with acquisition scripts.

## Troubleshooting

Aya was renamed to CohereLabs/aya_dataset. Its dataset-viewer filter sometimes returns HTTP 500; retries and cached source rows are supported. No available Odia Aya rows were found with the queried language codes. UI code-mixed/Odia support remains unproven.

## Verification

Inspect provenance JSONs and compare prompt_fingerprint sets before training. Do not sum cache, domain seed subsets, raw and derived exports as unique training examples.

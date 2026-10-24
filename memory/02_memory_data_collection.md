# Stage 02 — Data collection

Implemented `scripts/data/download_datasets.py`, `scripts/data/create_seed_data.py`, and `scripts/data/verify_data.py`. Seed data is reproducible and includes education, healthcare, agriculture, public-service, Hindi, Bengali, Tamil, Telugu, and Hinglish records.

Verified continuation state (2026-10-24): downloaded capped source samples to `data/raw/sft/alpaca_hindi.jsonl` (100 records) and `data/raw/sft/aya_dataset.jsonl` (100 records). `verify_data.py` reports 220 raw rows because its aggregate includes the six seed rows as well as their domain-subset files; this is not 220 unique source examples.

The current source samples do not establish the target Indic-language coverage claim: 100 rows have unknown language and most Aya rows are non-Indic. Filter and inspect provenance/licences before training. Run commands through the Nix runtime wrapper.

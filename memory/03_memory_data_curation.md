# Stage 03 — Curation

Implemented CuratorKIT configurations for cleaning, QA generation, multi-turn generation, and DPO preference generation, plus `run_curation.py` and `merge_curated_data.py`.

The runner requires an SGLang model endpoint on port 30000 and fails before issuing work if unavailable. It merges valid SFT rows to `data/curated/final_sft_train.jsonl` and valid DPO triplets to `data/preference/multilingual_pref.jsonl`. Dataset counts are produced only after an actual curation run.

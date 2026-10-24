# Stage 05 — DPO

Implemented `configs/aligntune/dpo_config.yaml` and `scripts/training/run_dpo.py`. The configuration trains from the SFT output with beta 0.1 and 1e-5 learning rate, consuming `data/preference/multilingual_pref.jsonl`.

`run_dpo.py` validates data and SFT output before starting. Merge the final adapter through `merge_lora.py`; record observed SFT/DPO comparisons after execution.

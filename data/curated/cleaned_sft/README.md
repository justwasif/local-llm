---
tags:
- curatorkit
configs:
- config_name: sft_messages
  default: true
  description: curatorkit sft_messages export, sha256 451592b37257de9db5ce1ce3f4e502633596d8a2a7e604ef6d7547750fab0743
  data_files:
  - split: train
    path: sft_messages.jsonl
---

# Dataset Card: indic_aya_clean

## Summary

- **Run timestamp**: 2026-10-24T19:29:53.621995Z
- **Wall clock time**: 0.2s
- **Pipeline config hash**: `fc03d2591a2af2e7`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 216 | 216 | — | 0 | 216 |
| SchemaGate | 216 | 216 | 0 | 0 | 216 |
| TextCleaner | 216 | 216 | — | — | 216 |
| ExactDeduplicator | 216 | 216 | — | — | 216 |
| MessagesExporter | — | 216 | — | — | 216 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 216/0 passed (N/A), 0 rejected
- **SchemaGate**: 216/216 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.22
- **python**: 3.12.3

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_messages` | train (216) | `messages`: list of {role, content}; role is system, user or assistant |

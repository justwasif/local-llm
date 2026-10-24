---
tags:
- curatorkit
configs:
- config_name: dpo
  default: true
  description: curatorkit dpo export, sha256 612ff4b8ff027b1a768aff58aa05066faab4eb687f07bc72c976902bfc7c0ddb
  data_files:
  - split: train
    path: dpo.jsonl
---

# Dataset Card: source_anchored_preferences

## Summary

- **Run timestamp**: 2026-10-24T19:42:36.798819Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `e35040d856311b70`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 7 | 7 | — | 0 | 7 |
| SchemaGate | 7 | 7 | 0 | 0 | 7 |
| ExactDeduplicator | 7 | 7 | — | — | 7 |
| DPOExporter | — | 7 | — | — | 7 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 7/0 passed (N/A), 0 rejected
- **SchemaGate**: 7/7 passed (100.0%), 0 rejected


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
| `dpo` | train (7) | `prompt`: string, or list of {role, content}<br>`chosen`: string, or list of {role, content}<br>`rejected`: string, or list of {role, content} |

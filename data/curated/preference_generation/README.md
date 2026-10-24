---
tags:
- curatorkit
configs:
- config_name: dpo
  default: true
  description: curatorkit dpo export, sha256 4de3bc27d52551caa4876e19ff69c93718786ebdcb16ce8b95f85bd0eda137e3
  data_files:
  - split: train
    path: dpo.jsonl
---

# Dataset Card: indic_preferences

## Summary

- **Run timestamp**: 2026-10-24T19:34:22.625450Z
- **Wall clock time**: 51.5s
- **Pipeline config hash**: `15e50f433c1e622c`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 8 | 8 | — | 0 | 8 |
| SchemaGate | 8 | 8 | 0 | 0 | 8 |
| TextCleaner | 8 | 8 | — | — | 8 |
| ExactDeduplicator | 8 | 8 | — | — | 8 |
| PreferenceGenerationTask | 8 | 8 | — | 0 | 8 |
| DPOExporter | — | 8 | — | — | 8 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 8/0 passed (N/A), 0 rejected
- **SchemaGate**: 8/8 passed (100.0%), 0 rejected
- **PreferenceGenerationTask**: 8/8 passed (100.0%), 0 rejected


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
| `dpo` | train (8) | `prompt`: string, or list of {role, content}<br>`chosen`: string, or list of {role, content}<br>`rejected`: string, or list of {role, content} |

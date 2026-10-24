---
tags:
- curatorkit
configs:
- config_name: sft_messages
  default: true
  description: curatorkit sft_messages export, sha256 550275432619c2121fe6b1f2b96ed0a71702c043e32694838d39d74a40bdfc27
  data_files:
  - split: train
    path: sft_messages.jsonl
---

# Dataset Card: grounded_indic_qa

## Summary

- **Run timestamp**: 2026-10-24T19:31:55.174757Z
- **Wall clock time**: 74.1s
- **Pipeline config hash**: `e2ce11e06472c776`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 6 | 6 | — | 0 | 6 |
| SchemaGate | 6 | 6 | 0 | 0 | 6 |
| TextCleaner | 6 | 6 | — | — | 6 |
| ExactDeduplicator | 6 | 6 | — | — | 6 |
| QAGenerationTask | 6 | 10 | — | 0 | 10 |
| HallucinationGate | 10 | 2 | 0 | 8 | 2 |
| MessagesExporter | — | 2 | — | — | 2 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 6/0 passed (N/A), 0 rejected
- **SchemaGate**: 6/6 passed (100.0%), 0 rejected
- **QAGenerationTask**: 10/6 passed (166.7%), 0 rejected
- **HallucinationGate**: 2/10 passed (20.0%), 8 rejected


**Rejection reasons:**
  - `hallucination_contract_failed:0.50`: 8

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
| `sft_messages` | train (2) | `messages`: list of {role, content}; role is system, user or assistant |

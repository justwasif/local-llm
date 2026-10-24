# Dataset Card: indic_multiturn

## Summary

- **Run timestamp**: 2026-10-24T19:33:01.218795Z
- **Wall clock time**: 35.7s
- **Pipeline config hash**: `446be4e3401cc0d2`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 6 | 6 | — | 0 | 6 |
| SchemaGate | 6 | 6 | 0 | 0 | 6 |
| TextCleaner | 6 | 6 | — | — | 6 |
| ExactDeduplicator | 6 | 6 | — | — | 6 |
| MultiTurnTask | 6 | 6 | — | 0 | 6 |
| HallucinationGate | 6 | 4 | 0 | 2 | 4 |
| MessagesExporter | — | 4 | — | — | 4 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 6/0 passed (N/A), 0 rejected
- **SchemaGate**: 6/6 passed (100.0%), 0 rejected
- **MultiTurnTask**: 6/6 passed (100.0%), 0 rejected
- **HallucinationGate**: 4/6 passed (66.7%), 2 rejected


**Rejection reasons:**
  - `hallucination_contract_failed:0.50`: 2

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

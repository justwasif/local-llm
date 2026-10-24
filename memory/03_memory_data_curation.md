# Stage 03 - CuratorKIT

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

{"cleaned_sft": {"JSONLReader": {"output_count": 216, "rejected_count": 0}, "SchemaGate": {"input_count": 216, "output_count": 216, "probe_recovered": 0, "rejected_count": 0}, "TextCleaner": {"input_count": 216, "output_count": 216}, "ExactDeduplicator": {"input_count": 216, "output_count": 216}, "MessagesExporter": {"exported_count": 216}}, "qa_generation": {"JSONLReader": {"output_count": 6, "rejected_count": 0}, "SchemaGate": {"input_count": 6, "output_count": 6, "probe_recovered": 0, "rejected_count": 0}, "TextCleaner": {"input_count": 6, "output_count": 6}, "ExactDeduplicator": {"input_count": 6, "output_count": 6}, "QAGenerationTask": {"input_count": 6, "output_count": 10, "rejected_count": 0}, "HallucinationGate": {"input_count": 10, "output_count": 2, "probe_recovered": 0, "rejected_count": 8}, "MessagesExporter": {"exported_count": 2}}, "multiturn_generation": {"JSONLReader": {"output_count": 6, "rejected_count": 0}, "SchemaGate": {"input_count": 6, "output_count": 6, "probe_recovered": 0, "rejected_count": 0}, "TextCleaner": {"input_count": 6, "output_count": 6}, "ExactDeduplicator": {"input_count": 6, "output_count": 6}, "MultiTurnTask": {"input_count": 6, "output_count": 6, "rejected_count": 0}, "HallucinationGate": {"input_count": 6, "output_count": 4, "probe_recovered": 0, "rejected_count": 2}, "MessagesExporter": {"exported_count": 4}}, "preference_generation": {"JSONLReader": {"output_count": 8, "rejected_count": 0}, "SchemaGate": {"input_count": 8, "output_count": 8, "probe_recovered": 0, "rejected_count": 0}, "TextCleaner": {"input_count": 8, "output_count": 8}, "ExactDeduplicator": {"input_count": 8, "output_count": 8}, "PreferenceGenerationTask": {"input_count": 8, "output_count": 8, "rejected_count": 0}, "DPOExporter": {"exported_count": 8}}, "reviewed_preferences": {"JSONLReader": {"output_count": 7, "rejected_count": 0}, "SchemaGate": {"input_count": 7, "output_count": 7, "probe_recovered": 0, "rejected_count": 0}, "ExactDeduplicator": {"input_count": 7, "output_count": 7}, "DPOExporter": {"exported_count": 7}}}

## File manifest

configs/curatorkit/*.yaml; scripts/data/run_curation.py; scripts/data/merge_curated_data.py; scripts/training/format_sft_data.py; tests/test_toolkit_inputs.py; data/curated/<pipeline>/{manifest.json,checksums.txt,lexsi_provenance.json,rejected.jsonl}.

## How it works

Use the published PipelineConfig schema. Input schema checks run BEFORE generation. Generators use real SGLang. The grounding judge accepted two factual errors: Bengali triangle geometry and a Tamil dengue passage misread as a wedding. Source-bound agent review excludes both using configs/synthetic_review.json and checks raw export hashes. Seven DPO chosen answers use handwritten references paired with reviewed weaker generations. Merge only canonical accepted exports, deduplicate prompts and trim unanswered trailing user turns. Final training data: 220 SFT conversations and seven preferences; no independent expert review.

## Configuration

SGLang base http://127.0.0.1:30000/v1, served model local-llm. Grounding threshold 0.6, judge failures rejected. Generated examples and same-model judgements are not independent factual/safety review.

## Dependencies

Current environments; frozen source records; SGLang must finish warm-up before LLM stages. Clean-only curation can run without an endpoint.

## Troubleshooting

High-level CuratorConfig YAML is not the CLI PipelineConfig schema. Pretrain reader consumes text, not a renamed output field. JSONLReader.read() returns (samples, rejected). Empty accepted exports cause run_curation.py to return failure. Do not weaken the grounding gate to hide rejected generations.

## Verification

Run curation and inspect every manifest stage count and rejection reason. Three toolkit integration tests exercise actual source readers/schema gates before generation. Merge/formatted outputs must be nonempty and disjoint from frozen evaluation prompts.

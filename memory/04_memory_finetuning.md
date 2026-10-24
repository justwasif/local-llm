# Stage 04 - AlignTune SFT

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

{"status": "completed", "toolkit": "aligntune", "versions": {"aligntune": "0.1.28", "torch": "2.11.0", "transformers": "5.20.2", "trl": "1.7.1", "peft": "0.21.2", "bitsandbytes": "0.50.2"}, "model": "Qwen/Qwen2.5-1.5B-Instruct", "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "dataset_count": 220, "dataset_sha256": "e504f9724bb205fa69bd994b6c30843b6a3c11a38195fe1d8c4662ddeb37d975", "duration_seconds": 137.6016662120819, "peak_allocated_vram_bytes": 2950186496, "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "global_steps": 55, "adapter_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_sft", "limitations": "Small local proof-of-pipeline; held-out benchmark comparison required before claiming improvement."}. Complete loss history is in the linked run report; final mean loss: 1.3475896987048062

## File manifest

scripts/training/run_local_sft.py; scripts/training/merge_local_adapter.py; configs/aligntune/local_1_5b_sft.yaml; outputs/local_1_5b_sft; results/reports/local_sft_run.json.

## How it works

Use AlignTune's published trainer factory with its TRL backend and PEFT 4-bit LoRA. Save toolkit provenance, adapter/tokenizer, trainer state, loss history, global steps, elapsed time and peak allocated VRAM. Merge using AlignTune PEFTMerger.

## Configuration

Explicit local Qwen2.5-1.5B-Instruct profile: sequence length 384, batch 1, accumulation 4, one epoch, LoRA rank 8/alpha 16, NF4 double quantization, BF16. SFT LR 1e-4; DPO LR 1e-5, beta 0.1. The 7B production config is separate and unexecuted here.

## Dependencies

SFT needs curated nonempty messages. DPO needs merged SFT weights and at least two distinct valid chosen/rejected pairs. Stop serving before either GPU training stage.

## Troubleshooting

The 7B profile is not suitable for this 8 GB GPU. Native package paths outside site-packages caused dill MonthDayNano serialization failure; the standard cache layout fixes it. Merging adds legacy RoPE/dtype/tokenizer metadata for Transformers 4.x serving without changing weight tensors. DPO toolkit result.total_steps is an unreliable zero; trainer.state.global_step records the actual two optimizer steps. Do not treat these small preference data as a safety guarantee.

## Verification

Require results/reports/local_sft_run.json status completed, positive optimizer steps, finite losses and a reloadable adapter. Compare held-out outputs through SGLang after merging.

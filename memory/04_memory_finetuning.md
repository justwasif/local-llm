# Stage 04 — SFT

Implemented `configs/aligntune/sft_config.yaml`, `format_sft_data.py`, `run_sft.py`, and `merge_lora.py`. The SFT plan uses Qwen2.5-7B-Instruct, QLoRA rank 32, alpha 64, three epochs, 2e-4 learning rate, and 2048 token sequences.

Format first, then start SFT. Training metrics, GPU use, checkpoints, and validation outputs are intentionally pending real hardware execution.

## 2026-10-24 smoke run

On the available RTX 4050 (6 GB), a Qwen2.5-1.5B 4-bit QLoRA smoke run completed against eight reviewed multilingual examples. The adapter is at `outputs/demo_1_5b_lora`; mean training loss was `2.5680`. Loading the adapter and generating Hindi, Bengali, and Hinglish replies completed successfully. Output quality was poor, as expected from eight examples and one epoch, so this must not be treated as a benchmark result.

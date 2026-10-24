# ⚖️ Stage 2.2: DPO Alignment — Direct Preference Optimization

This stage focuses on aligning the SFT-tuned model with human preferences to improve output quality, reduce hallucinations, and ensure better linguistic nuance in low-resource Indic languages.

## 🎯 Goal
Transform the SFT model into a preference-aligned model using **AlignTune** (powered by Unsloth/TRL). The goal is to ensure the model chooses the most natural, accurate, and helpful response among multiple candidates.

---

## 🛠️ Implementation Steps

### 1. Preference Dataset Preparation
DPO requires a dataset of triplets: `(prompt, chosen, rejected)`.
- **Source**: Use **CuratorKIT** to generate preference pairs or utilize existing multilingual preference datasets (e.g., UltraFeedback translated or curated).
- **Format**: JSONL
  ```json
  {"prompt": "What is the best way to grow wheat in Punjab?", "chosen": "The best way is... [Detailed, accurate, polite]", "rejected": "Just plant seeds. [Vague, incorrect, or robotic]"}
  ```
- **Path**: `data/preference/multilingual_pref.jsonl`

### 2. AlignTune DPO Configuration
Create a DPO configuration file in `configs/aligntune/dpo_config.yaml`.

**Recommended Parameters:**
- **Base Model**: The SFT model output from Stage 04 (`outputs/sft_model`).
- **Beta**: 0.1 (Standard starting point for DPO).
- **Learning Rate**: 5e-6 to 1e-5 (DPO requires a much lower LR than SFT).
- **Max Sequence Length**: 2048.
- **Optimizer**: AdamW with 4-bit quantization (via Unsloth).

### 3. Execute DPO Training
Run the training using the AlignTune CLI:
```bash
aligntune train --config configs/aligntune/dpo_config.yaml --method dpo
```

### 4. Model Merging & Export
Merge the DPO LoRA adapter back into the SFT base model to create the final aligned model.
```bash
aligntune merge \
    --base-model outputs/sft_model/final \
    --adapter outputs/dpo_model/checkpoint-final \
    --output outputs/merged_model
```

---

## ✅ Verification Checklist
- [ ] Preference dataset `data/preference/multilingual_pref.jsonl` exists and is correctly formatted.
- [ ] DPO training completes without Loss divergence (Loss should stabilize).
- [ ] Merged model is saved in `outputs/merged_model`.
- [ ] Sample test: The model prefers "chosen" style responses over "rejected" ones in Indic languages.

## 🧠 Memory Note
After completing this stage, create `memory/05_memory_dpo_alignment.md` documenting:
- The preference dataset sources.
- The final DPO hyperparameters used.
- Comparison of SFT vs DPO outputs for a few sample prompts.

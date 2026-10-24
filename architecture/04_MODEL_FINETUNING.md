# 🎯 Stage 4: Model Fine-Tuning with AlignTune (SFT)

> **Agent Instructions**: Complete ALL steps in this file, verify each one, then write `memory/04_memory_finetuning.md` before proceeding to `05_DPO_ALIGNMENT.md`.

> **Prerequisite**: Stage 3 (Data Curation) must be complete. Verify `data/curated/final_sft_train.jsonl` exists.

---

## 4.1 Overview

AlignTune is Lexsi Labs' fine-tuning toolkit built on **Unsloth** and **TRL**. It supports:
- **SFT** (Supervised Fine-Tuning) — what we do in this stage
- **DPO/ORPO** — next stage (05)
- **GRPO** — reinforcement learning (optional)
- **LoRA/QLoRA** — parameter-efficient fine-tuning

We will fine-tune **Qwen2.5-7B-Instruct** on our curated multilingual data using QLoRA (4-bit quantized base model + LoRA adapters).

---

## 4.2 Install AlignTune

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate

# Install AlignTune
pip install aligntune

# Verify
aligntune info
aligntune diagnose
```

---

## 4.3 Stop SGLang Server (Free GPU Memory)

> [!WARNING]
> Fine-tuning requires maximum GPU memory. Stop the SGLang server from Stage 3 before training.

```bash
# Kill any running SGLang server
pkill -f "sglang.launch_server" || true
# Wait for GPU memory to be freed
sleep 10
nvidia-smi  # Verify GPU memory is free
```

---

## 4.4 Create Data Formatting Script

Create `scripts/training/format_sft_data.py`:

```python
"""
Format curated data into AlignTune-compatible format.
Converts to ChatML / Messages format for Qwen2.5.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from utils import load_jsonl, save_jsonl, setup_logging, load_config, PROJECT_ROOT

logger = setup_logging("format_sft", "format_sft.log")
config = load_config()

SYSTEM_PROMPT = """You are a helpful, accurate, and knowledgeable multilingual AI assistant. 
You can communicate fluently in Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, 
Malayalam, Punjabi, Odia, and Hinglish (code-mixed Hindi-English).
You specialize in education, healthcare, agriculture, and public services for Indian citizens.
Always respond in the same language the user writes in. Be concise, accurate, and helpful."""


def format_alpaca_to_messages(item):
    """Convert Alpaca format to ChatML messages format."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    
    if not instruction and not output_text:
        return None
    
    # Combine instruction and input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    
    if not user_content.strip() or not output_text.strip():
        return None
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": output_text.strip()}
        ]
    }


def format_sharegpt_to_messages(item):
    """Convert ShareGPT format to ChatML messages format."""
    conversations = item.get("conversations", [])
    if not conversations:
        # Try messages format directly
        messages = item.get("messages", [])
        if messages:
            # Ensure system prompt is present
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            return {"messages": messages}
        return None
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        
        if role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
    
    if len(messages) < 3:  # Need at least system + user + assistant
        return None
    
    return {"messages": messages}


def format_text_to_messages(item):
    """Convert raw text to instruction format for continued pretraining."""
    text = item.get("text", "")
    if not text or len(text) < 50:
        return None
    
    language = item.get("language", "en")
    domain = item.get("domain", "general")
    
    # Create a reading comprehension style instruction
    prompts = {
        "hi": "नीचे दिए गए पाठ को पढ़ें और उसका सारांश बताएं:",
        "bn": "নীচের পাঠ্যটি পড়ুন এবং এর সারাংশ দিন:",
        "ta": "கீழே உள்ள உரையை படித்து சுருக்கமாக கூறுங்கள்:",
        "te": "కింది వచనాన్ని చదివి సారాంశం చెప్పండి:",
        "mr": "खालील मजकूर वाचा आणि त्याचा सारांश सांगा:",
        "en": "Read the following text and provide a summary:",
    }
    
    instruction = prompts.get(language, prompts["en"])
    
    # Truncate text if too long
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n{text}"},
            {"role": "assistant", "content": text[:500]}  # Use beginning as summary
        ]
    }


def main():
    logger.info("=" * 60)
    logger.info("📝 Formatting SFT Data for AlignTune")
    logger.info("=" * 60)
    
    input_path = PROJECT_ROOT / config['paths']['data_curated'] / "final_sft_train.jsonl"
    output_path = PROJECT_ROOT / config['paths']['data_curated'] / "sft_messages_train.jsonl"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run data curation (Stage 3) first!")
        sys.exit(1)
    
    raw_data = load_jsonl(str(input_path))
    logger.info(f"Loaded {len(raw_data)} raw samples")
    
    formatted_data = []
    skipped = 0
    
    for item in raw_data:
        formatted = None
        
        # Try different format conversions
        if "messages" in item:
            formatted = format_sharegpt_to_messages(item)
        elif "instruction" in item or "output" in item:
            formatted = format_alpaca_to_messages(item)
        elif "conversations" in item:
            formatted = format_sharegpt_to_messages(item)
        elif "text" in item:
            formatted = format_text_to_messages(item)
        
        if formatted:
            # Add metadata
            formatted["language"] = item.get("language", item.get("language_code", "unknown"))
            formatted["domain"] = item.get("domain", "general")
            formatted["source"] = item.get("source", "unknown")
            formatted_data.append(formatted)
        else:
            skipped += 1
    
    logger.info(f"Formatted: {len(formatted_data)} samples")
    logger.info(f"Skipped: {skipped} samples (malformed or too short)")
    
    # Save formatted data
    save_jsonl(formatted_data, str(output_path))
    logger.info(f"✅ Saved to: {output_path}")
    
    # Language distribution
    lang_counts = {}
    for item in formatted_data:
        lang = item.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    logger.info(f"Language distribution: {dict(sorted(lang_counts.items()))}")


if __name__ == "__main__":
    main()
```

---

## 4.5 Create AlignTune SFT Config

Create `configs/aligntune/sft_config.yaml`:

```yaml
# AlignTune SFT Config — Qwen2.5-7B-Instruct Multilingual Fine-Tuning
algo: sft

model:
  name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  backend: unsloth          # Use Unsloth for 2x speed + 60% less memory
  max_seq_length: 2048      # Balance between context length and memory
  torch_dtype: "bfloat16"

# QLoRA Configuration
peft:
  use_lora: true
  r: 32                     # Higher rank for multilingual (more capacity)
  lora_alpha: 64             # 2x rank is common
  target_modules: "all-linear"  # Apply LoRA to all linear layers
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

# Dataset
datasets:
  - name: "data/curated/sft_messages_train.jsonl"
    format: "messages"       # ChatML format with messages field

# Training Hyperparameters
train:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 4 * 4 = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  optim: "adamw_8bit"       # 8-bit Adam for memory efficiency

# Logging & Saving
logging:
  output_dir: "outputs/sft_model"
  logging_steps: 10
  save_strategy: "steps"
  save_steps: 200
  save_total_limit: 3
  report_to: "none"         # Set to "wandb" if using Weights & Biases
  run_name: "localllm-sft-multilingual"

# Evaluation during training
eval:
  eval_strategy: "steps"
  eval_steps: 200
  per_device_eval_batch_size: 4
```

---

## 4.6 Create Training Launch Script

Create `scripts/training/run_sft.py`:

```python
"""
Launch SFT training using AlignTune.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_logging, load_config, PROJECT_ROOT

logger = setup_logging("sft_training", "sft_training.log")


def run_sft_training():
    """Run SFT training via AlignTune CLI."""
    config_path = str(PROJECT_ROOT / "configs" / "aligntune" / "sft_config.yaml")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting SFT Training")
    logger.info(f"   Config: {config_path}")
    logger.info("=" * 60)
    
    # Method 1: AlignTune CLI
    try:
        result = subprocess.run(
            ["aligntune", "train", "--config", config_path],
            cwd=str(PROJECT_ROOT),
            text=True,
            timeout=86400  # 24 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ SFT training completed successfully!")
        else:
            logger.error(f"❌ SFT training failed with exit code {result.returncode}")
            # Try Python API fallback
            run_sft_python_api()
            
    except FileNotFoundError:
        logger.warning("'aligntune' CLI not found. Using Python API...")
        run_sft_python_api()
    except subprocess.TimeoutExpired:
        logger.error("⏰ Training timed out after 24 hours")


def run_sft_python_api():
    """Fallback: Run SFT using AlignTune Python API."""
    logger.info("Attempting SFT via Python API...")
    
    try:
        from aligntune.core.backend_factory import create_sft_trainer
        
        trainer = create_sft_trainer(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_name=str(PROJECT_ROOT / "data" / "curated" / "sft_messages_train.jsonl"),
            backend="unsloth",
            num_epochs=3,
            batch_size=4,
            learning_rate=2e-4,
            max_seq_length=2048,
            output_dir=str(PROJECT_ROOT / "outputs" / "sft_model"),
        )
        
        trainer.train()
        logger.info("✅ SFT training completed via Python API!")
        
    except Exception as e:
        logger.error(f"❌ Python API also failed: {e}")
        logger.info("Falling back to direct Unsloth/TRL training...")
        run_sft_direct()


def run_sft_direct():
    """Last resort: Run SFT using Unsloth + TRL directly."""
    logger.info("Running SFT with direct Unsloth + TRL...")
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
        import json
        
        # Load model with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        # Load dataset
        data_path = str(PROJECT_ROOT / "data" / "curated" / "sft_messages_train.jsonl")
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        # Format for training
        def format_messages(example):
            messages = example.get("messages", [])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}
        
        dataset = dataset.map(format_messages, remove_columns=dataset.column_names)
        
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.05, seed=42)
        
        # Training config
        training_config = SFTConfig(
            output_dir=str(PROJECT_ROOT / "outputs" / "sft_model"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            max_grad_norm=1.0,
            bf16=True,
            optim="adamw_8bit",
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=200,
            max_seq_length=2048,
            dataset_text_field="text",
            packing=True,  # Pack short sequences together for efficiency
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_config,
            train_dataset=split["train"],
            eval_dataset=split["test"],
        )
        
        # Train!
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save
        logger.info("Saving model...")
        trainer.save_model(str(PROJECT_ROOT / "outputs" / "sft_model" / "final"))
        tokenizer.save_pretrained(str(PROJECT_ROOT / "outputs" / "sft_model" / "final"))
        
        # Log metrics
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        logger.info(f"Training time: {train_result.metrics.get('train_runtime', 0):.0f}s")
        
        logger.info("✅ SFT training completed via direct Unsloth + TRL!")
        
    except Exception as e:
        logger.error(f"❌ Direct training also failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    run_sft_training()
```

---

## 4.7 Run SFT Training

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate

# Step 1: Format data
python scripts/training/format_sft_data.py

# Step 2: Verify formatted data
wc -l data/curated/sft_messages_train.jsonl
head -1 data/curated/sft_messages_train.jsonl | python -m json.tool

# Step 3: Run SFT training
python scripts/training/run_sft.py

# Step 4: Verify output
ls -la outputs/sft_model/
ls -la outputs/sft_model/final/
```

---

## 4.8 Merge LoRA Adapter (Post-Training)

Create `scripts/training/merge_lora.py`:

```python
"""
Merge LoRA adapter weights back into the base model.
Produces a full model that can be served without adapter loading.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import setup_logging, load_config, PROJECT_ROOT

logger = setup_logging("merge_lora", "merge_lora.log")
config = load_config()


def merge_with_unsloth():
    """Merge using Unsloth (fastest method)."""
    from unsloth import FastLanguageModel
    
    adapter_path = str(PROJECT_ROOT / "outputs" / "sft_model" / "final")
    merged_path = str(PROJECT_ROOT / "outputs" / "merged_model" / "sft_merged")
    
    logger.info(f"Loading adapter from: {adapter_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    logger.info("Merging LoRA weights into base model...")
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",  # Save as float16
    )
    
    logger.info(f"✅ Merged model saved to: {merged_path}")
    return merged_path


def merge_with_aligntune():
    """Merge using AlignTune CLI."""
    import subprocess
    
    result = subprocess.run([
        "aligntune", "merge",
        "--base-model", config['model']['base_model'],
        "--adapter", str(PROJECT_ROOT / "outputs" / "sft_model" / "final"),
        "--output", str(PROJECT_ROOT / "outputs" / "merged_model" / "sft_merged"),
    ], cwd=str(PROJECT_ROOT), text=True)
    
    if result.returncode == 0:
        logger.info("✅ Merged via AlignTune CLI")
    else:
        logger.error(f"❌ AlignTune merge failed: {result.returncode}")


def main():
    logger.info("=" * 60)
    logger.info("🔀 Merging LoRA Adapter with Base Model")
    logger.info("=" * 60)
    
    try:
        merge_with_unsloth()
    except Exception as e:
        logger.warning(f"Unsloth merge failed: {e}")
        try:
            merge_with_aligntune()
        except Exception as e2:
            logger.error(f"All merge methods failed: {e2}")


if __name__ == "__main__":
    main()
```

```bash
# Run the merge (after training completes)
python scripts/training/merge_lora.py
```

---

## 4.9 Quick Validation

Create `scripts/training/quick_test_sft.py`:

```python
"""
Quick test of the fine-tuned model before proceeding to DPO.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import setup_logging, PROJECT_ROOT

logger = setup_logging("quick_test", "quick_test.log")


def test_model():
    from unsloth import FastLanguageModel
    
    model_path = str(PROJECT_ROOT / "outputs" / "sft_model" / "final")
    
    logger.info(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    test_prompts = [
        # Hindi
        {"role": "user", "content": "भारत में मानसून कब आता है और यह कृषि के लिए क्यों महत्वपूर्ण है?"},
        # Bengali
        {"role": "user", "content": "ডায়াবেটিস কী এবং এটি কীভাবে নিয়ন্ত্রণ করা যায়?"},
        # Tamil
        {"role": "user", "content": "இந்தியாவின் கல்வி முறை பற்றி சுருக்கமாக விளக்கவும்."},
        # Hinglish
        {"role": "user", "content": "Bhai mujhe apna Aadhaar card update karna hai, kaise karu?"},
        # Telugu
        {"role": "user", "content": "వ్యవసాయంలో సేంద్రీయ ఎరువుల ప్రాముఖ్యత ఏమిటి?"},
    ]
    
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are a helpful multilingual AI assistant."},
            prompt,
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        lang = prompt["content"][:20]
        logger.info(f"\n{'='*40}")
        logger.info(f"Query ({lang}...): {prompt['content']}")
        logger.info(f"Response: {response}")
    
    logger.info("\n✅ Quick test completed!")


if __name__ == "__main__":
    test_model()
```

```bash
python scripts/training/quick_test_sft.py
```

---

## 4.10 Write Memory File

> [!IMPORTANT]
> **Agent**: Create `memory/04_memory_finetuning.md` with:
> 1. All scripts and config files created
> 2. Training hyperparameters used (lr, epochs, batch size, LoRA rank, etc.)
> 3. Training metrics (loss curves, final loss, runtime)
> 4. Model paths (adapter, merged)
> 5. Quick test results for each language
> 6. GPU memory usage during training
> 7. Any issues encountered and resolutions
> 8. How to resume training from a checkpoint

---

## ➡️ Next: [`05_DPO_ALIGNMENT.md`](./05_DPO_ALIGNMENT.md)

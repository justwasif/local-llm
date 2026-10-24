# 🧹 Stage 3: Data Curation with CuratorKIT

> **Agent Instructions**: Complete ALL steps in this file, verify each one, then write `memory/03_memory_data_curation.md` before proceeding to `04_MODEL_FINETUNING.md`.

> **Prerequisite**: Stage 2 (Data Collection) must be complete. Verify by checking `memory/02_memory_data_collection.md` exists.

---

## 3.1 Overview

CuratorKIT is Lexsi Labs' data curation toolkit. It uses **SGLang** internally via LiteLLM for:
- Synthetic data generation (QA pairs, multi-turn conversations)
- Data quality filtering with hallucination gates
- Provenance tracking
- Export to trainer-ready formats (Alpaca, ShareGPT, Messages)

**Pipeline**: Raw Data → CuratorKIT → Clean, augmented, trainer-ready datasets

---

## 3.2 Install CuratorKIT

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate

# Install CuratorKIT with all extras
pip install "curatorkit[all]"

# Verify installation
python -c "import curatorkit; print('CuratorKIT installed successfully')"
```

---

## 3.3 Start SGLang Server (Required by CuratorKIT)

CuratorKIT uses SGLang as its LLM backend. Start the server first:

```bash
# Start SGLang server with the base model
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85

# Wait for server to be ready (check with):
curl -fsS http://localhost:30000/v1/models
```

> [!WARNING]
> Keep this server running in a **separate terminal/background process** throughout the curation stage. All CuratorKIT operations depend on it.

---

## 3.4 Create CuratorKIT Pipeline Configs

### 3.4.1 QA Generation Pipeline

Create `configs/curatorkit/qa_generation.yaml`:

```yaml
# CuratorKIT Pipeline: Generate QA pairs from seed data
# Uses SGLang-served Qwen2.5-7B as generator and judge

dataset:
  name: "data/raw/seeds/all_seeds.jsonl"
  max_samples: null  # Use all seeds

hygiene:
  secrets_detection: true
  pii_masking: true
  toxicity_filter: true

generation_task: "qa"
num_questions: 5  # Generate 5 QA pairs per seed document

generator_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.7
  max_tokens: 2048

judge_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.1

system_prompt: >
  You are an expert multilingual AI assistant and data creator. 
  Generate accurate question-answer pairs strictly grounded in the provided source text.
  IMPORTANT: Generate the question and answer in the SAME LANGUAGE as the source text.
  If the source is in Hindi, generate in Hindi. If in Bengali, generate in Bengali.
  If the source is in Hinglish (mixed Hindi-English), generate in Hinglish.
  Cover the key information in the source text thoroughly.

hallucination_threshold: 0.75
reward_threshold: 0.70
dedup: "embedding"
enable_reward_refiner: true

export_formats:
  - "alpaca"
  - "sharegpt"
  - "messages"

output_dir: "data/curated/qa_generation"
```

### 3.4.2 Multi-Turn Conversation Pipeline

Create `configs/curatorkit/multiturn_generation.yaml`:

```yaml
# CuratorKIT Pipeline: Generate multi-turn conversations for domain topics

dataset:
  name: "data/raw/seeds/all_seeds.jsonl"
  max_samples: null

hygiene:
  secrets_detection: true
  pii_masking: false
  toxicity_filter: true

generation_task: "multi_turn"
num_questions: 3  # 3 turns per conversation

generator_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.8
  max_tokens: 2048

judge_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.1

system_prompt: >
  You are a helpful multilingual assistant specializing in education, healthcare, 
  agriculture, and public services for Indian citizens. Generate natural multi-turn 
  conversations in the SAME LANGUAGE as the source text. The conversations should 
  be realistic — as if a citizen is asking questions and getting helpful answers.
  Include follow-up questions that deepen understanding.

hallucination_threshold: 0.70
reward_threshold: 0.65
dedup: "minhash"
enable_reward_refiner: true

export_formats:
  - "sharegpt"
  - "messages"

output_dir: "data/curated/multiturn_generation"
```

### 3.4.3 Preference Data Generation Pipeline

Create `configs/curatorkit/preference_generation.yaml`:

```yaml
# CuratorKIT Pipeline: Generate DPO preference pairs

dataset:
  name: "data/raw/seeds/all_seeds.jsonl"
  max_samples: null

hygiene:
  secrets_detection: true
  pii_masking: true
  toxicity_filter: true

generation_task: "preference"

generator_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.8
  max_tokens: 2048

judge_llm:
  model: "openai/Qwen/Qwen2.5-7B-Instruct"
  api_base: "http://localhost:30000/v1"
  api_key: "EMPTY"
  temperature: 0.1

system_prompt: >
  Generate preference pairs in the same language as the source text.
  The "chosen" response should be accurate, helpful, and well-structured.
  The "rejected" response should contain subtle errors, be less helpful, or
  have quality issues that make it clearly inferior.

hallucination_threshold: 0.80
reward_threshold: 0.75
dedup: "embedding"
enable_reward_refiner: false

export_formats:
  - "dpo"

output_dir: "data/curated/preference_generation"
```

### 3.4.4 Raw SFT Data Cleaning Pipeline

Create `configs/curatorkit/clean_sft_data.yaml`:

```yaml
# CuratorKIT Pipeline: Clean and deduplicate raw SFT datasets

dataset:
  name: "data/raw/sft/"  # Process all files in this directory
  max_samples: null

hygiene:
  secrets_detection: true
  pii_masking: true
  toxicity_filter: true

# No generation — just clean and deduplicate
dedup: "minhash"
clean: true

export_formats:
  - "alpaca"
  - "messages"

output_dir: "data/curated/cleaned_sft"
```

---

## 3.5 Create Curation Runner Script

Create `scripts/data/run_curation.py`:

```python
"""
Run all CuratorKIT pipelines to produce curated training data.
Requires SGLang server running on port 30000.
"""

import os
import sys
import subprocess
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_logging, load_config, PROJECT_ROOT

logger = setup_logging("curation", "curation.log")
config = load_config()

SGLANG_URL = f"http://localhost:{config['sglang']['port']}/v1/models"


def check_sglang_server():
    """Verify SGLang server is running."""
    try:
        resp = requests.get(SGLANG_URL, timeout=5)
        if resp.status_code == 200:
            logger.info("✅ SGLang server is running")
            return True
    except requests.ConnectionError:
        pass
    logger.error("❌ SGLang server is NOT running on port 30000")
    logger.error("   Start it with: python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000")
    return False


def run_pipeline(config_path: str, name: str):
    """Run a single CuratorKIT pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Running pipeline: {name}")
    logger.info(f"   Config: {config_path}")
    logger.info(f"{'='*60}")
    
    full_path = str(PROJECT_ROOT / config_path)
    
    try:
        result = subprocess.run(
            ["curatorkit", "run", full_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per pipeline
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Pipeline '{name}' completed successfully")
            if result.stdout:
                logger.info(f"   Output: {result.stdout[-500:]}")  # Last 500 chars
        else:
            logger.error(f"❌ Pipeline '{name}' failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"   Error: {result.stderr[-500:]}")
                
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Pipeline '{name}' timed out after 1 hour")
    except FileNotFoundError:
        logger.warning(f"   'curatorkit' CLI not found. Trying Python API...")
        run_pipeline_python(full_path, name)


def run_pipeline_python(config_path: str, name: str):
    """Fallback: Run pipeline using CuratorKIT Python API."""
    try:
        from curatorkit import Curator, CuratorConfig
        import yaml
        
        with open(config_path) as f:
            pipeline_config = yaml.safe_load(f)
        
        curator = Curator(CuratorConfig(**pipeline_config))
        result = curator.run()
        result.print_summary()
        logger.info(f"✅ Pipeline '{name}' completed via Python API")
        
    except Exception as e:
        logger.error(f"❌ Pipeline '{name}' failed via Python API: {e}")


def main():
    """Run all curation pipelines."""
    logger.info("🚀 Starting CuratorKIT Data Curation")
    
    # Check SGLang server
    if not check_sglang_server():
        logger.error("Cannot proceed without SGLang server. Exiting.")
        sys.exit(1)
    
    pipelines = [
        ("configs/curatorkit/clean_sft_data.yaml", "Clean Raw SFT Data"),
        ("configs/curatorkit/qa_generation.yaml", "QA Generation"),
        ("configs/curatorkit/multiturn_generation.yaml", "Multi-Turn Conversations"),
        ("configs/curatorkit/preference_generation.yaml", "Preference Pairs (DPO)"),
    ]
    
    for config_path, name in pipelines:
        run_pipeline(config_path, name)
        time.sleep(5)  # Brief pause between pipelines
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All curation pipelines complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 3.6 Create Data Merging Script

Create `scripts/data/merge_curated_data.py`:

```python
"""
Merge all curated datasets into final training-ready files.
Produces:
  - data/curated/final_sft_train.jsonl  (for SFT training)
  - data/curated/final_dpo_train.jsonl  (for DPO training)
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
from utils import load_jsonl, save_jsonl, setup_logging, load_config, PROJECT_ROOT

logger = setup_logging("merge_data", "merge_data.log")
config = load_config()

CURATED_DIR = PROJECT_ROOT / config['paths']['data_curated']


def merge_sft_data():
    """Merge all SFT data into a single training file."""
    logger.info("Merging SFT data...")
    
    all_data = []
    
    # 1. Cleaned raw SFT data
    cleaned_dir = CURATED_DIR / "cleaned_sft"
    if cleaned_dir.exists():
        for f in cleaned_dir.rglob("*.jsonl"):
            data = load_jsonl(str(f))
            all_data.extend(data)
            logger.info(f"  Added {len(data)} samples from {f.name}")
    
    # 2. Generated QA data
    qa_dir = CURATED_DIR / "qa_generation"
    if qa_dir.exists():
        for f in qa_dir.rglob("*.jsonl"):
            if "rejected" not in f.name:  # Skip rejected samples
                data = load_jsonl(str(f))
                all_data.extend(data)
                logger.info(f"  Added {len(data)} samples from {f.name}")
    
    # 3. Multi-turn conversation data
    mt_dir = CURATED_DIR / "multiturn_generation"
    if mt_dir.exists():
        for f in mt_dir.rglob("*.jsonl"):
            if "rejected" not in f.name:
                data = load_jsonl(str(f))
                all_data.extend(data)
                logger.info(f"  Added {len(data)} samples from {f.name}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Save
    output_path = CURATED_DIR / "final_sft_train.jsonl"
    save_jsonl(all_data, str(output_path))
    logger.info(f"\n✅ Merged SFT data: {len(all_data)} samples → {output_path}")
    
    # Print language distribution
    lang_counts = {}
    for item in all_data:
        lang = item.get("language", item.get("language_code", "unknown"))
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    logger.info(f"   Language distribution: {dict(sorted(lang_counts.items()))}")
    
    return len(all_data)


def merge_preference_data():
    """Merge all preference data for DPO training."""
    logger.info("\nMerging Preference (DPO) data...")
    
    all_data = []
    
    # 1. Generated preference pairs
    pref_gen_dir = CURATED_DIR / "preference_generation"
    if pref_gen_dir.exists():
        for f in pref_gen_dir.rglob("*.jsonl"):
            if "rejected" not in f.name:
                data = load_jsonl(str(f))
                all_data.extend(data)
                logger.info(f"  Added {len(data)} pairs from {f.name}")
    
    # 2. Raw downloaded preference data (if cleaned)
    raw_pref_dir = PROJECT_ROOT / config['paths']['data_raw'] / "preference"
    if raw_pref_dir.exists():
        for f in raw_pref_dir.rglob("*.jsonl"):
            data = load_jsonl(str(f))
            all_data.extend(data)
            logger.info(f"  Added {len(data)} pairs from {f.name}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Save
    output_path = PROJECT_ROOT / config['paths']['data_preference'] / "final_dpo_train.jsonl"
    save_jsonl(all_data, str(output_path))
    logger.info(f"\n✅ Merged DPO data: {len(all_data)} pairs → {output_path}")
    
    return len(all_data)


def main():
    logger.info("=" * 60)
    logger.info("🔀 Merging Curated Data")
    logger.info("=" * 60)
    
    sft_count = merge_sft_data()
    dpo_count = merge_preference_data()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data merge complete!")
    logger.info(f"   SFT training samples: {sft_count}")
    logger.info(f"   DPO preference pairs: {dpo_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 3.7 Run the Full Curation Pipeline

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate

# Step 1: Ensure SGLang server is running (in a separate terminal)
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 30000 &

# Step 2: Wait for server to be ready
sleep 60
curl -fsS http://localhost:30000/v1/models

# Step 3: Run curation pipelines
python scripts/data/run_curation.py

# Step 4: Merge curated data
python scripts/data/merge_curated_data.py
```

---

## 3.8 Verify Curated Data

```bash
# Check output files exist
ls -la data/curated/
ls -la data/curated/final_sft_train.jsonl
ls -la data/preference/final_dpo_train.jsonl

# Check sample counts
wc -l data/curated/final_sft_train.jsonl
wc -l data/preference/final_dpo_train.jsonl

# Preview a few samples
head -3 data/curated/final_sft_train.jsonl | python -m json.tool
```

---

## 3.9 Write Memory File

> [!IMPORTANT]
> **Agent**: Create `memory/03_memory_data_curation.md` with:
> 1. All CuratorKIT config files created and their purpose
> 2. All scripts created (`run_curation.py`, `merge_curated_data.py`)
> 3. SGLang server config used (model, port, etc.)
> 4. Number of samples produced by each pipeline
> 5. Final merged dataset statistics (total samples, language distribution)
> 6. CuratorKIT provenance artifacts generated
> 7. Any pipeline failures and how they were handled
> 8. How to re-run specific pipelines if needed

---

## ➡️ Next: [`04_MODEL_FINETUNING.md`](./04_MODEL_FINETUNING.md)

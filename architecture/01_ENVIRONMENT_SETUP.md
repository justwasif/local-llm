# 📦 Stage 1: Environment Setup

> **Agent Instructions**: Complete ALL steps in this file, verify each one, then write `memory/01_memory_environment.md` before proceeding to `02_DATA_COLLECTION.md`.

---

## 1.1 System Requirements Check

```bash
# Check GPU availability and VRAM
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version (need 3.10+)
python3 --version
```

> [!IMPORTANT]
> Minimum requirements:
> - NVIDIA GPU with ≥ 16GB VRAM (24GB recommended)
> - CUDA 12.1+ installed
> - Python 3.10 or 3.11 or 3.12
> - 50GB free disk space (for model + datasets)

---

## 1.2 Create Project Structure

```bash
# From the project root: /home/just_wasif/Documents/github/local llm/
cd "/home/just_wasif/Documents/github/local llm"

# Create all directories
mkdir -p data/{raw,curated,preference,eval}
mkdir -p configs/{curatorkit,aligntune,auditkit,sglang}
mkdir -p scripts/{data,training,eval,serve}
mkdir -p src/gui
mkdir -p outputs/{sft_model,dpo_model,merged_model}
mkdir -p results/{benchmarks,safety,reports}
mkdir -p memory
mkdir -p logs
```

---

## 1.3 Python Virtual Environment

```bash
cd "/home/just_wasif/Documents/github/local llm"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## 1.4 Install Dependencies

Create `requirements.txt`:

```txt
# === Core: SGLang (MANDATORY) ===
sglang[all]

# === Lexsi Labs Toolkit ===
curatorkit
aligntune
auditkit

# === Training Dependencies ===
unsloth
trl>=0.8.0
peft>=0.10.0
bitsandbytes>=0.43.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.28.0
tokenizers>=0.15.0

# === Evaluation ===
lm-eval>=0.4.0
rouge-score
sacrebleu
nltk

# === Data Processing ===
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
jsonlines
sentencepiece
protobuf

# === GUI ===
gradio>=4.0.0

# === Utilities ===
pyyaml
tqdm
wandb
huggingface-hub>=0.20.0
safetensors
```

```bash
# Install all dependencies
pip install -r requirements.txt

# If SGLang install fails due to flashinfer, try:
# pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

---

## 1.5 HuggingFace Authentication

```bash
# Login to HuggingFace (needed for model downloads)
huggingface-cli login
# Enter your token when prompted

# Or set as environment variable
export HF_TOKEN="your_token_here"
```

---

## 1.6 Create Project Configuration

Create `configs/project_config.yaml`:

```yaml
# Project Configuration — LocalLLM Multilingual
project:
  name: "localllm-multilingual"
  version: "1.0.0"
  problem_statement: "PS4 - Low-resource and multilingual languages"
  hackathon: "SYNTAX ERROR 2026"

model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  model_family: "qwen2"
  max_seq_length: 4096
  quantization: "4bit"  # For training; full precision for eval

target_languages:
  primary:
    - code: "hi"
      name: "Hindi"
    - code: "bn"
      name: "Bengali"
    - code: "ta"
      name: "Tamil"
    - code: "te"
      name: "Telugu"
    - code: "mr"
      name: "Marathi"
  secondary:
    - code: "gu"
      name: "Gujarati"
    - code: "kn"
      name: "Kannada"
    - code: "ml"
      name: "Malayalam"
    - code: "pa"
      name: "Punjabi"
    - code: "or"
      name: "Odia"
  code_mixed:
    - code: "hi-en"
      name: "Hinglish"
    - code: "bn-en"
      name: "Banglish"

domains:
  - education
  - healthcare
  - agriculture
  - public_services

sglang:
  host: "0.0.0.0"
  port: 30000
  mem_fraction_static: 0.85

paths:
  data_raw: "data/raw"
  data_curated: "data/curated"
  data_preference: "data/preference"
  data_eval: "data/eval"
  output_sft: "outputs/sft_model"
  output_dpo: "outputs/dpo_model"
  output_merged: "outputs/merged_model"
  results: "results"
```

---

## 1.7 Create Utility Script

Create `scripts/utils.py`:

```python
"""
Shared utility functions for the LocalLLM project.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        log_path = PROJECT_ROOT / "logs" / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def load_config(config_path: str = "configs/project_config.yaml") -> Dict:
    """Load project configuration."""
    full_path = PROJECT_ROOT / config_path
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def save_jsonl(data: List[Dict], output_path: str):
    """Save a list of dicts to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(input_path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


def get_gpu_info() -> Dict:
    """Get GPU information."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
    }


def format_chat_message(role: str, content: str) -> Dict:
    """Format a single chat message."""
    return {"role": role, "content": content}


def format_chat_conversation(
    system_prompt: str,
    user_message: str,
    assistant_response: str
) -> List[Dict]:
    """Format a full chat conversation for training."""
    messages = []
    if system_prompt:
        messages.append(format_chat_message("system", system_prompt))
    messages.append(format_chat_message("user", user_message))
    messages.append(format_chat_message("assistant", assistant_response))
    return messages
```

---

## 1.8 Create Verification Script

Create `scripts/verify_setup.py`:

```python
"""
Verify that the environment is correctly set up.
Run this after completing 01_ENVIRONMENT_SETUP.md.
"""

import sys
import os
import importlib

def check_python_version():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, f"Python 3.10+ required, got {v.major}.{v.minor}"
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")

def check_gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")

def check_packages():
    packages = {
        "sglang": "SGLang",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "trl": "TRL",
        "gradio": "Gradio",
        "yaml": "PyYAML",
        "pandas": "Pandas",
        "accelerate": "Accelerate",
    }
    
    for pkg, name in packages.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")

def check_directories():
    from pathlib import Path
    root = Path(__file__).parent.parent
    
    required_dirs = [
        "data/raw", "data/curated", "data/preference", "data/eval",
        "configs", "scripts", "src/gui",
        "outputs/sft_model", "outputs/dpo_model", "outputs/merged_model",
        "results/benchmarks", "results/safety", "results/reports",
        "memory", "logs"
    ]
    
    for d in required_dirs:
        path = root / d
        if path.exists():
            print(f"✅ Directory: {d}")
        else:
            print(f"❌ Directory missing: {d}")

def check_config():
    from pathlib import Path
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "project_config.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded: {config['project']['name']}")
    print(f"   Base model: {config['model']['base_model']}")
    print(f"   Languages: {len(config['target_languages']['primary'])} primary, "
          f"{len(config['target_languages']['secondary'])} secondary")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 LocalLLM Environment Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu),
        ("Package Installation", check_packages),
        ("Directory Structure", check_directories),
        ("Project Config", check_config),
    ]
    
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        try:
            check_fn()
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)
```

---

## 1.9 Verify Setup

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate
python scripts/verify_setup.py
```

---

## 1.10 Write Memory File

> [!IMPORTANT]
> **Agent**: After completing all steps above, create `memory/01_memory_environment.md` with:
> 1. All files created and their paths
> 2. What each file does
> 3. Package versions installed
> 4. GPU info detected
> 5. Any issues encountered and how they were resolved
> 6. Verification results

**Template for memory file:**
```markdown
# Memory: 01 — Environment Setup
## Date: [timestamp]
## Status: COMPLETED / PARTIAL

### Files Created
- `requirements.txt` — Python dependencies
- `configs/project_config.yaml` — Master project configuration
- `scripts/utils.py` — Shared utility functions (logging, config loading, JSONL I/O)
- `scripts/verify_setup.py` — Environment verification script

### Key Decisions
- Base model: Qwen/Qwen2.5-7B-Instruct
- Python version: X.XX
- CUDA version: XX.X
- GPU: [name] with [X]GB VRAM

### Dependencies on This Stage
- ALL subsequent stages depend on this setup
- `scripts/utils.py` is imported by all other scripts

### How to Verify
- Run: `python scripts/verify_setup.py`
- All checks should show ✅
```

---

## ➡️ Next: [`02_DATA_COLLECTION.md`](./02_DATA_COLLECTION.md)

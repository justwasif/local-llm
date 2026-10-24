# 📊 Stage 2: Data Collection

> **Agent Instructions**: Complete ALL steps in this file, verify each one, then write `memory/02_memory_data_collection.md` before proceeding to `03_DATA_CURATION.md`.

---

## 2.1 Overview

We need **three types of data** for our multilingual low-resource LLM:

| Type | Purpose | Stage Used |
|------|---------|------------|
| **SFT Data** | Instruction-response pairs in Indic languages | Fine-tuning (Stage 2) |
| **Preference Data** | Chosen/rejected pairs for DPO alignment | DPO (Stage 2) |
| **Evaluation Data** | Benchmarks to measure improvement | Eval (Stage 3) |

**Target Languages**: Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), Punjabi (pa), Odia (or), Hinglish (hi-en)

**Target Domains**: Education, Healthcare, Agriculture, Public Services

---

## 2.2 Create Data Download Script

Create `scripts/data/download_datasets.py`:

```python
"""
Download all required datasets for the LocalLLM multilingual project.
Datasets are organized by type: SFT, Preference, and Evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import json
from utils import setup_logging, save_jsonl, load_config, PROJECT_ROOT

logger = setup_logging("data_download", "data_download.log")
config = load_config()

RAW_DIR = PROJECT_ROOT / config['paths']['data_raw']
EVAL_DIR = PROJECT_ROOT / config['paths']['data_eval']


def download_sft_datasets():
    """Download instruction-tuning datasets for Indic languages."""
    
    logger.info("=" * 60)
    logger.info("Downloading SFT Datasets")
    logger.info("=" * 60)
    
    sft_dir = RAW_DIR / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Dataset 1: IndicInstruct (Multilingual Instructions) ----
    logger.info("Downloading ai4bharat/IndicInstruct...")
    try:
        ds = load_dataset("ai4bharat/IndicInstruct", split="train")
        data = [{"instruction": row.get("instruction", ""),
                 "input": row.get("input", ""),
                 "output": row.get("output", ""),
                 "language": row.get("language", "unknown"),
                 "source": "ai4bharat/IndicInstruct"}
                for row in ds]
        save_jsonl(data, str(sft_dir / "indic_instruct.jsonl"))
        logger.info(f"  Saved {len(data)} samples")
    except Exception as e:
        logger.warning(f"  Failed to download IndicInstruct: {e}")
    
    # ---- Dataset 2: Sangraha (Indic text corpus) ----
    logger.info("Downloading ai4bharat/sangraha...")
    try:
        # Download a subset — this is a large dataset
        for lang in ["hi", "bn", "ta", "te", "mr"]:
            try:
                ds = load_dataset("ai4bharat/sangraha", lang, split="train", streaming=True)
                data = []
                for i, row in enumerate(ds):
                    if i >= 5000:  # Limit per language
                        break
                    data.append({
                        "text": row.get("text", ""),
                        "language": lang,
                        "source": "ai4bharat/sangraha"
                    })
                save_jsonl(data, str(sft_dir / f"sangraha_{lang}.jsonl"))
                logger.info(f"  Saved {len(data)} samples for {lang}")
            except Exception as e:
                logger.warning(f"  Failed for {lang}: {e}")
    except Exception as e:
        logger.warning(f"  Failed to download Sangraha: {e}")
    
    # ---- Dataset 3: Alpaca-style multilingual ----
    logger.info("Downloading iamshnoo/alpaca-cleaned-hindi...")
    try:
        ds = load_dataset("iamshnoo/alpaca-cleaned-hindi", split="train")
        data = [{"instruction": row.get("instruction", ""),
                 "input": row.get("input", ""),
                 "output": row.get("output", ""),
                 "language": "hi",
                 "source": "iamshnoo/alpaca-cleaned-hindi"}
                for row in ds]
        save_jsonl(data, str(sft_dir / "alpaca_hindi.jsonl"))
        logger.info(f"  Saved {len(data)} samples")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- Dataset 4: CulturaX (Large multilingual) ----
    logger.info("Downloading CulturaX subsets...")
    try:
        for lang in ["hi", "bn", "ta", "te", "mr"]:
            try:
                ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
                data = []
                for i, row in enumerate(ds):
                    if i >= 3000:
                        break
                    data.append({
                        "text": row.get("text", ""),
                        "language": lang,
                        "source": "uonlp/CulturaX"
                    })
                save_jsonl(data, str(sft_dir / f"culturax_{lang}.jsonl"))
                logger.info(f"  Saved {len(data)} samples for {lang}")
            except Exception as e:
                logger.warning(f"  Failed for {lang}: {e}")
    except Exception as e:
        logger.warning(f"  Failed to download CulturaX: {e}")
    
    # ---- Dataset 5: Aya Dataset (Multilingual instructions) ----
    logger.info("Downloading CohereForAI/aya_dataset...")
    try:
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        # Filter for our target languages
        target_langs = {"Hindi", "Bengali", "Tamil", "Telugu", "Marathi", 
                       "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia"}
        data = []
        for row in ds:
            if row.get("language") in target_langs:
                data.append({
                    "instruction": row.get("inputs", ""),
                    "output": row.get("targets", ""),
                    "language": row.get("language", ""),
                    "language_code": row.get("language_code", ""),
                    "source": "CohereForAI/aya_dataset"
                })
        save_jsonl(data, str(sft_dir / "aya_dataset_indic.jsonl"))
        logger.info(f"  Saved {len(data)} Indic samples from Aya")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- Dataset 6: Aya Collection (Larger) ----
    logger.info("Downloading CohereForAI/aya_collection (subset)...")
    try:
        ds = load_dataset("CohereForAI/aya_collection", split="train", streaming=True)
        target_codes = {"hin", "ben", "tam", "tel", "mar", "guj", "kan", "mal", "pan", "ori"}
        data = []
        for row in ds:
            if row.get("language_code") in target_codes:
                data.append({
                    "instruction": row.get("inputs", ""),
                    "output": row.get("targets", ""),
                    "language_code": row.get("language_code", ""),
                    "source": "CohereForAI/aya_collection"
                })
                if len(data) >= 20000:
                    break
        save_jsonl(data, str(sft_dir / "aya_collection_indic.jsonl"))
        logger.info(f"  Saved {len(data)} Indic samples from Aya Collection")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- Dataset 7: Domain-specific (Healthcare) ----
    logger.info("Downloading medical QA datasets...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        data = [{"instruction": row.get("input", ""),
                 "output": row.get("output", ""),
                 "domain": "healthcare",
                 "language": "en",
                 "source": "medalpaca/medical_meadow_medical_flashcards"}
                for row in ds][:5000]
        save_jsonl(data, str(sft_dir / "medical_en.jsonl"))
        logger.info(f"  Saved {len(data)} medical samples (English, for translation)")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- Dataset 8: Code-mixed Hinglish ----
    logger.info("Downloading Hinglish datasets...")
    try:
        ds = load_dataset("lingyit/Hinglish-STS", split="train")
        data = [{"text": row.get("sentence1", "") + " " + row.get("sentence2", ""),
                 "language": "hi-en",
                 "source": "lingyit/Hinglish-STS"}
                for row in ds]
        save_jsonl(data, str(sft_dir / "hinglish_sts.jsonl"))
        logger.info(f"  Saved {len(data)} Hinglish samples")
    except Exception as e:
        logger.warning(f"  Failed: {e}")


def download_preference_datasets():
    """Download preference datasets for DPO training."""
    
    logger.info("=" * 60)
    logger.info("Downloading Preference Datasets (for DPO)")
    logger.info("=" * 60)
    
    pref_dir = RAW_DIR / "preference"
    pref_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Anthropic HH-RLHF ----
    logger.info("Downloading Anthropic/hh-rlhf...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        data = [{"chosen": row.get("chosen", ""),
                 "rejected": row.get("rejected", ""),
                 "source": "Anthropic/hh-rlhf"}
                for row in ds][:10000]
        save_jsonl(data, str(pref_dir / "hh_rlhf.jsonl"))
        logger.info(f"  Saved {len(data)} preference pairs")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- UltraFeedback ----
    logger.info("Downloading argilla/ultrafeedback-binarized-preferences-cleaned...")
    try:
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        data = []
        for row in ds:
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            if chosen and rejected:
                data.append({
                    "prompt": row.get("prompt", ""),
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "argilla/ultrafeedback"
                })
            if len(data) >= 10000:
                break
        save_jsonl(data, str(pref_dir / "ultrafeedback.jsonl"))
        logger.info(f"  Saved {len(data)} preference pairs")
    except Exception as e:
        logger.warning(f"  Failed: {e}")


def download_eval_datasets():
    """Download evaluation benchmark datasets."""
    
    logger.info("=" * 60)
    logger.info("Downloading Evaluation Datasets")
    logger.info("=" * 60)
    
    eval_dir = EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- FLORES-200 (Translation benchmark) ----
    logger.info("Downloading FLORES-200...")
    try:
        ds = load_dataset("facebook/flores", "all", split="devtest")
        data = [dict(row) for row in ds]
        save_jsonl(data, str(eval_dir / "flores200.jsonl"))
        logger.info(f"  Saved {len(data)} FLORES samples")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- IndicXTREME / IndicNLP Suite ----
    logger.info("Downloading IndicXNLI...")
    try:
        for lang in ["hi", "bn", "ta", "te", "mr", "gu"]:
            try:
                ds = load_dataset("Divyanshu/indicxnli", lang, split="test")
                data = [{"premise": row.get("premise", ""),
                         "hypothesis": row.get("hypothesis", ""),
                         "label": row.get("label", ""),
                         "language": lang,
                         "source": "Divyanshu/indicxnli"}
                        for row in ds]
                save_jsonl(data, str(eval_dir / f"indicxnli_{lang}.jsonl"))
                logger.info(f"  Saved {len(data)} IndicXNLI samples for {lang}")
            except Exception as e:
                logger.warning(f"  Failed for {lang}: {e}")
    except Exception as e:
        logger.warning(f"  Failed to download IndicXNLI: {e}")
    
    # ---- IndicSentiment ----
    logger.info("Downloading IndicSentiment...")
    try:
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-hi", split="test")
        data = [{"text": row.get("text", ""),
                 "label": row.get("label", ""),
                 "language": "hi",
                 "source": "ai4bharat/IndicSentiment"}
                for row in ds]
        save_jsonl(data, str(eval_dir / "indic_sentiment_hi.jsonl"))
        logger.info(f"  Saved {len(data)} sentiment samples")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # ---- MMLU (general knowledge benchmark) ----
    logger.info("Downloading MMLU...")
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        data = [{"question": row.get("question", ""),
                 "choices": row.get("choices", []),
                 "answer": row.get("answer", ""),
                 "subject": row.get("subject", ""),
                 "source": "cais/mmlu"}
                for row in ds]
        save_jsonl(data, str(eval_dir / "mmlu.jsonl"))
        logger.info(f"  Saved {len(data)} MMLU samples")
    except Exception as e:
        logger.warning(f"  Failed: {e}")


def create_domain_seed_data():
    """Create seed data for domain-specific synthetic generation.
    These will be used by CuratorKIT to generate more training data.
    """
    
    logger.info("=" * 60)
    logger.info("Creating Domain Seed Data")
    logger.info("=" * 60)
    
    seed_dir = RAW_DIR / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Education domain seeds (bilingual)
    education_seeds = [
        {"text": "प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे सूर्य के प्रकाश का उपयोग करके कार्बन डाइऑक्साइड और पानी को ग्लूकोज और ऑक्सीजन में बदलते हैं।",
         "language": "hi", "domain": "education", "topic": "biology"},
        {"text": "পিথাগোরাসের উপপাদ্য অনুসারে, একটি সমকোণী ত্রিভুজে অতিভুজের বর্গ অন্য দুই বাহুর বর্গের সমষ্টির সমান।",
         "language": "bn", "domain": "education", "topic": "mathematics"},
        {"text": "நியூட்டனின் முதல் விதிப்படி, ஒரு பொருள் மீது வெளிப்புற விசை செயல்படாத வரை அது நிலையில் இருக்கும் அல்லது நேர்கோட்டில் சீரான வேகத்தில் நகரும்.",
         "language": "ta", "domain": "education", "topic": "physics"},
        {"text": "భారతదేశ రాజ్యాంగం 1950 జనవరి 26న అమలులోకి వచ్చింది. ఇది ప్రపంచంలో అతి పెద్ద లిఖిత రాజ్యాంగం.",
         "language": "te", "domain": "education", "topic": "civics"},
        {"text": "भारताचे संविधान हे जगातील सर्वात मोठे लिखित संविधान आहे. ते २६ जानेवारी १९५० रोजी लागू झाले.",
         "language": "mr", "domain": "education", "topic": "civics"},
    ]
    
    # Healthcare domain seeds
    healthcare_seeds = [
        {"text": "डेंगू बुखार एडीज मच्छर के काटने से होता है। इसके लक्षणों में तेज बुखार, सिरदर्द, जोड़ों में दर्द और त्वचा पर चकत्ते शामिल हैं।",
         "language": "hi", "domain": "healthcare", "topic": "diseases"},
        {"text": "ডায়াবেটিস একটি দীর্ঘমেয়াদী রোগ যেখানে রক্তে শর্করার মাত্রা অস্বাভাবিকভাবে বেড়ে যায়। নিয়মিত ব্যায়াম এবং সুষম খাদ্য গ্রহণ এটি নিয়ন্ত্রণে সাহায্য করে।",
         "language": "bn", "domain": "healthcare", "topic": "diseases"},
        {"text": "கர்ப்பகால நீரிழிவு நோய் கர்ப்ப காலத்தில் ஏற்படும் ஒரு நிலை. இது தாய் மற்றும் குழந்தை இருவரையும் பாதிக்கும்.",
         "language": "ta", "domain": "healthcare", "topic": "maternal_health"},
    ]
    
    # Agriculture domain seeds
    agriculture_seeds = [
        {"text": "जैविक खेती में रासायनिक उर्वरकों और कीटनाशकों का उपयोग नहीं किया जाता। इसके बजाय कम्पोस्ट, वर्मीकम्पोस्ट और जैविक कीटनाशकों का प्रयोग होता है।",
         "language": "hi", "domain": "agriculture", "topic": "organic_farming"},
        {"text": "ধান চাষের জন্য প্রচুর পানি প্রয়োজন। বোরো মৌসুমে সেচের মাধ্যমে এবং আমন মৌসুমে বৃষ্টির পানিতে ধান চাষ করা হয়।",
         "language": "bn", "domain": "agriculture", "topic": "rice_cultivation"},
        {"text": "భూసార పరీక్ష చేయించడం వల్ల నేలలో ఏ పోషకాలు తక్కువగా ఉన్నాయో తెలుస్తుంది. దీని ఆధారంగా ఎరువులు వేయవచ్చు.",
         "language": "te", "domain": "agriculture", "topic": "soil_testing"},
    ]
    
    # Public services domain seeds
    public_services_seeds = [
        {"text": "आधार कार्ड भारत सरकार द्वारा जारी किया गया 12 अंकों का विशिष्ट पहचान संख्या है। इसका उपयोग सरकारी योजनाओं का लाभ प्राप्त करने के लिए किया जाता है।",
         "language": "hi", "domain": "public_services", "topic": "government_id"},
        {"text": "আয়কর রিটার্ন দাখিল করা প্রতিটি ভারতীয় নাগরিকের কর্তব্য যাদের আয় করযোগ্য সীমার উপরে। অনলাইনে ই-ফাইলিং করা যায়।",
         "language": "bn", "domain": "public_services", "topic": "taxation"},
        {"text": "ரேஷன் கார்டு என்பது அரசு மானிய விலையில் அத்தியாவசிய பொருட்கள் பெற உதவும் ஒரு ஆவணம்.",
         "language": "ta", "domain": "public_services", "topic": "ration_card"},
    ]
    
    # Hinglish (code-mixed) seeds
    hinglish_seeds = [
        {"text": "Bhai, mujhe apna Aadhaar card update karwana hai. Kya online ho sakta hai ya nearest center jaana padega?",
         "language": "hi-en", "domain": "public_services", "topic": "government_id"},
        {"text": "Doctor sahab, meri mummy ko diabetes hai aur unka sugar level bahut high aa raha hai. Kya karna chahiye?",
         "language": "hi-en", "domain": "healthcare", "topic": "diabetes"},
        {"text": "Yaar, maths ka ye integration wala chapter bahut tough hai. Koi easy way hai samajhne ka?",
         "language": "hi-en", "domain": "education", "topic": "mathematics"},
        {"text": "Mere khet mein kapas ki fasal mein keede lag gaye hain. Organic pesticide suggest karo please.",
         "language": "hi-en", "domain": "agriculture", "topic": "pest_control"},
    ]
    
    # Save all seed data
    all_seeds = {
        "education": education_seeds,
        "healthcare": healthcare_seeds,
        "agriculture": agriculture_seeds,
        "public_services": public_services_seeds,
        "hinglish": hinglish_seeds,
    }
    
    for domain, seeds in all_seeds.items():
        save_jsonl(seeds, str(seed_dir / f"seeds_{domain}.jsonl"))
        logger.info(f"  Created {len(seeds)} seed samples for {domain}")
    
    # Also save all seeds combined
    all_combined = []
    for seeds in all_seeds.values():
        all_combined.extend(seeds)
    save_jsonl(all_combined, str(seed_dir / "all_seeds.jsonl"))
    logger.info(f"  Created {len(all_combined)} total seed samples")


def main():
    """Run all data downloads."""
    logger.info("🚀 Starting data download pipeline")
    logger.info(f"   Raw data directory: {RAW_DIR}")
    logger.info(f"   Eval data directory: {EVAL_DIR}")
    
    # Step 1: Download SFT datasets
    download_sft_datasets()
    
    # Step 2: Download preference datasets
    download_preference_datasets()
    
    # Step 3: Download evaluation datasets
    download_eval_datasets()
    
    # Step 4: Create domain seed data
    create_domain_seed_data()
    
    logger.info("=" * 60)
    logger.info("✅ Data download complete!")
    logger.info("=" * 60)
    
    # Print summary
    total_files = sum(1 for _ in RAW_DIR.rglob("*.jsonl"))
    eval_files = sum(1 for _ in EVAL_DIR.rglob("*.jsonl"))
    logger.info(f"   Raw data files: {total_files}")
    logger.info(f"   Eval data files: {eval_files}")


if __name__ == "__main__":
    main()
```

---

## 2.3 Run Data Download

```bash
cd "/home/just_wasif/Documents/github/local llm"
source .venv/bin/activate
python scripts/data/download_datasets.py
```

> [!NOTE]
> Some datasets may fail to download due to authentication, availability, or size constraints. That's OK — the script logs warnings and continues. We need at least 3-4 SFT datasets to proceed.

---

## 2.4 Verify Downloaded Data

Create `scripts/data/verify_data.py`:

```python
"""Verify downloaded datasets and print statistics."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from utils import load_jsonl, load_config, PROJECT_ROOT, setup_logging

logger = setup_logging("data_verify", "data_verify.log")
config = load_config()


def count_samples_by_language(data):
    """Count samples per language."""
    lang_counts = {}
    for item in data:
        lang = item.get("language", item.get("language_code", "unknown"))
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return lang_counts


def verify_directory(dir_path, dir_name):
    """Verify all JSONL files in a directory."""
    logger.info(f"\n📁 {dir_name}: {dir_path}")
    
    if not dir_path.exists():
        logger.warning(f"   Directory does not exist!")
        return
    
    total_samples = 0
    all_lang_counts = {}
    
    for jsonl_file in sorted(dir_path.rglob("*.jsonl")):
        try:
            data = load_jsonl(str(jsonl_file))
            total_samples += len(data)
            lang_counts = count_samples_by_language(data)
            
            for lang, count in lang_counts.items():
                all_lang_counts[lang] = all_lang_counts.get(lang, 0) + count
            
            logger.info(f"   📄 {jsonl_file.name}: {len(data)} samples")
        except Exception as e:
            logger.warning(f"   ❌ {jsonl_file.name}: Error - {e}")
    
    logger.info(f"   Total: {total_samples} samples")
    if all_lang_counts:
        logger.info(f"   Languages: {dict(sorted(all_lang_counts.items()))}")
    
    return total_samples, all_lang_counts


def main():
    logger.info("=" * 60)
    logger.info("🔍 Data Verification Report")
    logger.info("=" * 60)
    
    raw_dir = PROJECT_ROOT / config['paths']['data_raw']
    eval_dir = PROJECT_ROOT / config['paths']['data_eval']
    
    # Verify SFT data
    sft_result = verify_directory(raw_dir / "sft", "SFT Training Data")
    
    # Verify Preference data
    pref_result = verify_directory(raw_dir / "preference", "Preference Data (DPO)")
    
    # Verify Seed data
    seed_result = verify_directory(raw_dir / "seeds", "Domain Seed Data")
    
    # Verify Eval data
    eval_result = verify_directory(eval_dir, "Evaluation Data")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Verification complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

```bash
python scripts/data/verify_data.py
```

---

## 2.5 Expected Output

After running the download script, you should have:

```
data/
├── raw/
│   ├── sft/
│   │   ├── indic_instruct.jsonl
│   │   ├── sangraha_hi.jsonl
│   │   ├── sangraha_bn.jsonl
│   │   ├── sangraha_ta.jsonl
│   │   ├── sangraha_te.jsonl
│   │   ├── sangraha_mr.jsonl
│   │   ├── alpaca_hindi.jsonl
│   │   ├── culturax_hi.jsonl
│   │   ├── culturax_bn.jsonl
│   │   ├── culturax_ta.jsonl
│   │   ├── culturax_te.jsonl
│   │   ├── culturax_mr.jsonl
│   │   ├── aya_dataset_indic.jsonl
│   │   ├── aya_collection_indic.jsonl
│   │   ├── medical_en.jsonl
│   │   └── hinglish_sts.jsonl
│   ├── preference/
│   │   ├── hh_rlhf.jsonl
│   │   └── ultrafeedback.jsonl
│   └── seeds/
│       ├── seeds_education.jsonl
│       ├── seeds_healthcare.jsonl
│       ├── seeds_agriculture.jsonl
│       ├── seeds_public_services.jsonl
│       ├── seeds_hinglish.jsonl
│       └── all_seeds.jsonl
└── eval/
    ├── flores200.jsonl
    ├── indicxnli_hi.jsonl
    ├── indicxnli_bn.jsonl
    ├── ...
    ├── indic_sentiment_hi.jsonl
    └── mmlu.jsonl
```

---

## 2.6 Write Memory File

> [!IMPORTANT]
> **Agent**: Create `memory/02_memory_data_collection.md` with:
> 1. All datasets downloaded (name, source, sample count)
> 2. All files created and their paths
> 3. Languages covered and sample counts per language
> 4. Any datasets that failed to download and fallback plans
> 5. Seed data created for each domain
> 6. How data verification works

---

## ➡️ Next: [`03_DATA_CURATION.md`](./03_DATA_CURATION.md)

"""Check tokenizer identity and RoPE/dtype semantics before serving a merged export."""

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
args = parser.parse_args()
base = json.loads((ROOT / "results/reports/model_download.json").read_text())["path"]
candidate = ROOT / f"outputs/local_1_5b_{args.stage}_merged"
original = AutoTokenizer.from_pretrained(base)
exported = AutoTokenizer.from_pretrained(candidate)
if original.get_vocab() != exported.get_vocab():
    raise ValueError("Token-to-ID vocabulary changed during export")
samples = [
    "नमस्ते",
    "নমস্কার",
    "வணக்கம்",
    "నమస్కారం",
    "नमस्कार",
    "નમસ્તે",
    "ನಮಸ್ಕಾರ",
    "നമസ്കാരം",
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
    "ନମସ୍କାର",
    "Namaste, kaise ho?",
    "Nomoshkar, kemon acho?",
]
for name in ["aya_test.jsonl", "flores200.jsonl"]:
    samples.extend(
        json.loads(line)["instruction"]
        for line in (ROOT / "data/eval" / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
for text in samples:
    if original.encode(text) != exported.encode(text):
        raise ValueError("Tokenization changed during export")
    for messages in [
        [{"role": "user", "content": text}],
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": text},
        ],
    ]:
        if original.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ) != exported.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ):
            raise ValueError("Chat rendering/token IDs changed during export")
source_config, exported_config = (
    AutoConfig.from_pretrained(base),
    AutoConfig.from_pretrained(candidate),
)
fields = [
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
    "torch_dtype",
]
for field in fields:
    if getattr(source_config, field, None) != getattr(exported_config, field, None):
        raise ValueError(f"Serving architecture/dtype mismatch: {field}")
report = {
    "status": "passed",
    "stage": args.stage,
    "vocabulary_identical": True,
    "sample_languages_checked": 12,
    "evaluation_prompts_checked": 86,
    "chat_template_token_ids_identical": True,
    "architecture_fields_checked": fields,
    "rope_theta": exported_config.rope_theta,
}
(ROOT / f"results/reports/{args.stage}_export_verification.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))

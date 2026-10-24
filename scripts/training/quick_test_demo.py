"""Load the saved 1.5B QLoRA adapter and run multilingual generation checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, configure_nix_cuda_runtime

configure_nix_cuda_runtime()

PROMPTS = [
    "प्रकाश संश्लेषण को दो वाक्यों में समझाइए।",
    "ডেঙ্গুর সতর্ক লক্ষণগুলি কী?",
    "Mere khet mein patton par daag hain, pehla safe step kya hai?",
]


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        print("CUDA is unavailable.", file=sys.stderr)
        return 2
    base = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = PROJECT_ROOT / "outputs/demo_1_5b_lora"
    if not adapter.exists():
        print("Adapter is missing; run run_demo_sft.py first.", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, device_map={"": 0}, dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    system = "You are a careful multilingual assistant for Indian public services. Reply in the user's language."
    for prompt in PROMPTS:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"PROMPT: {prompt}\nANSWER: {answer}\n---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

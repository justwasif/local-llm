"""Run a real QLoRA smoke fine-tune on Qwen2.5-1.5B for 6 GB GPUs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, configure_nix_cuda_runtime
configure_nix_cuda_runtime()

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); parser.add_argument("--epochs", type=int, default=1); args = parser.parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        print("CUDA is unavailable. Complete the CUDA PyTorch installation before training.", file=sys.stderr); return 2
    data_file = PROJECT_ROOT / "data/curated/demo_sft_messages.jsonl"
    if not data_file.exists():
        print("Demo data missing; run create_demo_training_data.py first.", file=sys.stderr); return 2
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    rows = load_dataset("json", data_files=str(data_file), split="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train(); losses = []
    for epoch in range(args.epochs):
        for index, row in enumerate(rows):
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
            result = model(**batch, labels=batch.input_ids)
            (result.loss / 4).backward()
            if (index + 1) % 4 == 0 or index + 1 == len(rows):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(result.loss.detach().cpu()))
            print(f"epoch={epoch + 1} sample={index + 1}/{len(rows)} loss={losses[-1]:.4f}", flush=True)
    output = PROJECT_ROOT / "outputs/demo_1_5b_lora"
    output.mkdir(parents=True, exist_ok=True); model.save_pretrained(output); tokenizer.save_pretrained(output)
    print(f"Saved adapter to {output}; mean_loss={sum(losses) / len(losses):.4f}")
    return 0
if __name__ == "__main__": raise SystemExit(main())

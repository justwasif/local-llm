# 📊 Stage 3.1: Evaluation & Benchmarking — AuditKIT

This stage focuses on measuring the model's performance against baseline benchmarks and ensuring the model is safe and helpful.

## 🎯 Goal
Use **AuditKIT** and **SGLang** to quantify the improvements in low-resource Indic languages and verify that the model meets the PS requirements for public-facing domains.

---

## 🛠️ Implementation Steps

### 1. Setup Evaluation Backend
AuditKIT uses SGLang as the inference engine. First, launch the SGLang server with the merged model:
```bash
python -m sglang.launch_server \
    --model-path outputs/merged_model \
    --host 0.0.0.0 \
    --port 30000
```

### 2. Configure Benchmarks
Create an evaluation configuration in `configs/auditkit/eval_config.yaml`.

**Target Benchmarks:**
- **MMLU (Multilingual)**: General knowledge.
- **FLORES-200**: Translation quality across Indic languages.
- **IndicXTREME / IndicNLP**: Domain-specific Indic benchmarks.
- **Custom Domain Evals**: Create a small set of Q&A pairs for Education, Healthcare, and Agriculture in target languages.

### 3. Run Evaluation
Execute the evaluation suite via AuditKIT:
```bash
auditkit eval \
    --model outputs/merged_model \
    --backend sglang \
    --config configs/auditkit/eval_config.yaml \
    --output results/benchmarks/eval_report.json
```

### 4. Safety & Hallucination Audit
Run safety tests to ensure the model doesn't produce harmful or biased content in multilingual settings:
```bash
auditkit safety \
    --model outputs/merged_model \
    --backend sglang \
    --tests toxicity,bias,hallucination \
    --output results/safety/safety_report.json
```

---

## ✅ Verification Checklist
- [ ] SGLang server is running and accessible.
- [ ] Evaluation reports generated in `results/benchmarks/` and `results/safety/`.
- [ ] Model outperforms the base Qwen2.5-7B-Instruct on target Indic benchmarks.
- [ ] Hallucination rate is within acceptable limits for public service use.

## 🧠 Memory Note
After completing this stage, create `memory/06_memory_evaluation.md` documenting:
- Final benchmark scores (BLEU, ROUGE, Accuracy).
- Comparison with base model performance.
- Any safety failures identified and how they were mitigated.

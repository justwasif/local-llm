# 🚀 Stage 3.2: SGLang Deployment — Serving & Optimization

This stage focuses on deploying the final merged model using the SGLang engine to achieve maximum throughput and minimum latency for the end-user.

## 🎯 Goal
Deploy a high-performance inference server that can handle multiple concurrent requests with low latency, optimized for consumer-grade GPUs (≤ 24GB VRAM).

---

## 🛠️ Implementation Steps

### 1. Server Optimization Configuration
Create a serving configuration in `configs/sglang/server_config.yaml`.

**Key Optimization Strategies:**
- **Quantization**: Use AWQ or GPTQ (via AlignTune export) to reduce VRAM usage if the 7B model is too heavy for the target GPU.
- **Tensor Parallelism (TP)**: Set `--tp 1` for single GPU, or higher if multi-GPU is available.
- **RadixAttention**: Enable to cache common system prompts and multilingual prefixes, significantly speeding up repeated queries.
- **Continuous Batching**: Ensure enabled for production-level throughput.

### 2. Launch the Production Server
Launch the server using the optimized settings:
```bash
python -m sglang.launch_server \
    --model-path outputs/merged_model \
    --quantization awq \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8
```

### 3. Performance Benchmarking
Verify the deployment's efficiency using SGLang's built-in tools:
```bash
python -m sglang.bench_serving \
    --backend sglang \
    --host localhost --port 30000 \
    --dataset-name random \
    --num-prompts 1000
```

---

## ✅ Verification Checklist
- [ ] Server launches and is accessible via OpenAI-compatible API (`/v1/chat/completions`).
- [ ] VRAM usage is stable and does not OOM under load.
- [ ] Latency (Time-To-First-Token) is minimized.
- [ ] Throughput (tokens/sec) is measured and documented.

## 🧠 Memory Note
After completing this stage, create `memory/07_memory_deployment.md` documenting:
- The final serving command and parameters.
- Measured throughput and latency benchmarks.
- Quantization method used (if any) and its impact on quality.

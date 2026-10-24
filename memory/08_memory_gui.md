# Stage 08 — GUI

Implemented `src/gui/app.py`: a Unicode-safe streaming Gradio chat UI with twelve language choices, four public-service domains, domain examples, safety disclaimers, and OpenAI-compatible SGLang integration.

Launch with `scripts/with_gpu_runtime.sh .venv/bin/python src/gui/app.py`. It defaults to `http://localhost:30000/v1`; use `SGLANG_BASE_URL` and `SGLANG_MODEL` to point it to another OpenAI-compatible endpoint. GUI construction was tested, but it has not been exercised against a live SGLang endpoint.

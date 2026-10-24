"""Exercise the running Gradio queue and real SGLang streaming, without browser claims."""

import json
import re
from pathlib import Path

import requests
from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]


def text(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(item.get("text", "") for item in content)


def main():
    url = "http://127.0.0.1:7860"
    requests.get(url, timeout=10).raise_for_status()
    client = Client(url, verbose=False)
    prompt = "नमस्ते! दो और दो कितने होते हैं? केवल एक छोटा वाक्य लिखें।"
    job = client.submit(prompt, [], "Hindi", "Education", api_name="/chat")
    updates = list(job)
    history = job.result()
    answer = text(history[-1])
    assert updates and answer.strip()
    assert text(history[-2]) == prompt
    assert "server unavailable" not in answer.lower()
    assert "unable to complete" not in answer.lower()
    assert re.search(r"[\u0900-\u097f]", answer), (
        "No Hindi script in requested response"
    )
    followup = "उसी उत्तर को अंकों में लिखें।"
    next_history = client.predict(
        followup, history, "Hindi", "Education", api_name="/chat"
    )
    assert next_history[:2] == history
    assert text(next_history[-2]) == followup and text(next_history[-1]).strip()
    empty = client.predict("  ", next_history, "Hindi", "Education", api_name="/chat")
    assert empty == next_history
    cleared = client.predict(api_name="/lambda_2")
    assert cleared == ([], "") or cleared == [[], ""]
    model = requests.get("http://127.0.0.1:30000/model_info", timeout=10).json()
    assert model["model_path"] == str(ROOT / "outputs/local_1_5b_dpo_merged")
    report = {
        "status": "passed",
        "url": url,
        "loaded_model_path": model["model_path"],
        "checks": [
            "HTTP page",
            "real queued streaming",
            "Unicode input preserved",
            "Hindi-script response",
            "follow-up history",
            "empty message",
            "clear",
        ],
        "stream_updates": len(updates),
        "prompt": prompt,
        "response": answer,
        "followup_response": text(next_history[-1]),
        "arithmetic_answer_marker_present": bool(re.search(r"(?:\b4\b|चार)", answer)),
        "quality_limit": "Arithmetic marker presence is a limited diagnostic, not a factual correctness score. Functional success must not be read as model-quality success. The initial pre-fix GUI responses are preserved in gui_verification_before_history_fix.json.",
        "visual_browser_check": "blocked: no browser connector; computer-use tool rejected Chrome access as not approved",
        "limitations": "Functional API smoke test; not independent language-quality or visual-layout verification.",
    }
    (ROOT / "results/reports/gui_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

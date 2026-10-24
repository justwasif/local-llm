"""Write CuratorKIT 0.1.22 CLI schemas, replacing legacy high-level configs."""

from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
for filename, task, name in [
    ("clean_sft_data", None, "indic_aya_clean"),
    ("qa_generation", "qa", "grounded_indic_qa"),
    ("multiturn_generation", "multiturn", "indic_multiturn"),
    ("preference_generation", "preference", "indic_preferences"),
]:
    reader = {
        "type": "jsonl",
        "path": "data/raw/sft/indic_aya_train.jsonl",
        "format": "alpaca",
    }
    config = {
        "name": name,
        "readers": [reader],
        "gates": [{"type": "schema", "min_tokens": 2, "max_tokens": 2048}],
        "normalizers": [{"type": "text_cleaner"}, {"type": "exact_dedup"}],
        "exporters": [{"type": "messages"}],
    }
    if task:
        reader.update(path="data/raw/seeds/all_seeds.jsonl", format="pretrain")
        config["gates"][0]["required_fields"] = ["output"]
        config["llm"] = {
            "model": "openai/local-llm",
            "api_base": "http://127.0.0.1:30000/v1",
            "api_key": "EMPTY",
            "temperature": 0.2,
            "max_tokens": 768,
            "concurrency": 1,
            "timeout": 120,
            "max_retries": 1,
        }
        config["generators"] = [{"type": task}]
        if task == "qa":
            config["generators"][0]["num_questions"] = 2
            config["generators"][0]["prompt_template"] = (
                'Create exactly {num_questions} simple question-answer pairs grounded only in this source. Use the SAME language and code mixing as the source. Keep each answer to one short sentence. Source: {context}\nReturn ONLY a JSON array: [{{"question":"...", "answer":"..."}}]'
            )
        if task == "multiturn":
            config["generators"][0]["num_turns"] = 2
        if task != "preference":
            config["gates"].append(
                {
                    "type": "hallucination",
                    "hallucination_threshold": 0.6,
                    "on_judge_error": "reject",
                }
            )
        else:
            config["readers"] = [
                {
                    "type": "jsonl",
                    "path": "data/raw/seeds/preference_seeds.jsonl",
                    "format": "alpaca",
                }
            ]
            config["generators"][0]["prompt_template"] = (
                'Create a preference pair. Use the SAME language or code mixing as the instruction. The chosen answer must be specific, concise and supported by the reference. The rejected answer must be vague or incomplete, while remaining factually correct. Do not add facts. Instruction: {instruction}\nReference: {context_section}\nReturn ONLY JSON: {{"chosen":"...", "rejected":"..."}}'
            )
            config["gates"][0]["required_fields"] = ["instruction", "output"]
            config["exporters"] = [{"type": "dpo"}]
    (root / "configs/curatorkit" / f"{filename}.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

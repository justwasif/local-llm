"""Preserve Qwen2 RoPE/dtype semantics when serving Transformers 5 exports with 4.x."""

import argparse
import json
from pathlib import Path


def compatible_config(config):
    result = dict(config)
    changes = {}
    if config.get("model_type") != "qwen2":
        raise ValueError("This compatibility export is verified only for Qwen2")
    rope = config.get("rope_parameters")
    if rope:
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Scaled RoPE needs separate compatibility validation")
        if "rope_theta" not in config:
            changes["rope_theta"] = rope["rope_theta"]
            result["rope_theta"] = rope["rope_theta"]
    if "dtype" in config and "torch_dtype" not in config:
        changes["torch_dtype"] = config["dtype"]
        result["torch_dtype"] = config["dtype"]
    return result, changes


def ensure_compatible(path):
    path = Path(path)
    config_file = path / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    result, changes = compatible_config(config)
    tokenizer_file = path / "tokenizer_config.json"
    tokenizer_changes = []
    if tokenizer_file.exists():
        tokenizer = json.loads(tokenizer_file.read_text(encoding="utf-8"))
        if isinstance(tokenizer.get("extra_special_tokens"), list):
            tokenizer["additional_special_tokens"] = tokenizer.pop(
                "extra_special_tokens"
            )
            tokenizer_changes.append(
                "extra_special_tokens list renamed to additional_special_tokens for Transformers 4.x"
            )
            tokenizer_file.write_text(
                json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    if changes or tokenizer_changes:
        config_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        metadata = path / "serving_compatibility.json"
        report = (
            json.loads(metadata.read_text())
            if metadata.exists()
            else {"added_fields": {}}
        )
        report.update(
            reason="Preserve model and token-ID semantics across Transformers 5 training / 4 serving",
            weights_modified=False,
        )
        report["added_fields"].update(changes)
        report["tokenizer_metadata_changes"] = tokenizer_changes
        metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Serving compatibility fields: {changes}; tokenizer metadata: {tokenizer_changes}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    ensure_compatible(parser.parse_args().model_path)

from types import SimpleNamespace

import auditkit as ak

from scripts.eval.local_evaluation import (
    checkpoint_identity,
    evaluation_model,
    referral_screen,
)


def test_custom_safety_metric_uses_explicit_direction():
    sample = ak.Sample(
        input="Question",
        actual_output="Contact a doctor",
        metadata={"expected_markers": ["doctor"]},
    )
    metric = ak.scorer(
        referral_screen, direction=ak.Direction.MAXIMIZE, name="referral_keyword_screen"
    )
    result = ak.evaluate([sample], model="precomputed", scorers=metric)
    assert result.headline["referral_keyword_screen"] == 1.0
    assert not result.errors


def test_safety_marker_matching_is_case_insensitive():
    sample = SimpleNamespace(metadata={"expected_markers": ["doctor"]})
    assert referral_screen(sample, "Ask a DOCTOR") == 1.0
    assert referral_screen(sample, "") == 0.0


def test_dataset_fingerprinting_works_in_native_cache():
    from datasets import Dataset

    data = Dataset.from_dict({"text": ["नमस्ते", "নমস্কার"]})
    assert len(data) == 2
    assert data._fingerprint


def test_evaluation_cache_separates_checkpoint_weights(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"baseline")
    first = evaluation_model("http://127.0.0.1:30000/v1", checkpoint_identity(tmp_path))
    weights.write_bytes(b"trained")
    second = evaluation_model(
        "http://127.0.0.1:30000/v1", checkpoint_identity(tmp_path)
    )
    assert first.identity() != second.identity()
    assert first._model_name == second._model_name == "local-llm"

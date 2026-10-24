import pytest

from scripts.serve.ensure_legacy_config import compatible_config


def test_transformers5_export_preserves_legacy_rope_and_dtype():
    original = {
        "model_type": "qwen2",
        "dtype": "bfloat16",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    result, changes = compatible_config(original)
    assert result["rope_theta"] == 1000000.0
    assert result["torch_dtype"] == "bfloat16"
    assert "rope_theta" not in original
    assert changes == {"rope_theta": 1000000.0, "torch_dtype": "bfloat16"}


def test_scaled_rope_is_not_silently_converted():
    with pytest.raises(ValueError, match="Scaled RoPE"):
        compatible_config(
            {"model_type": "qwen2", "rope_parameters": {"rope_type": "yarn"}}
        )

import pytest
import torch

from steering_recovery.runtime import (
    dtype_name,
    is_gpt2_small_model,
    resolve_model_dtype,
)


def test_gpt2_small_always_resolves_to_float32():
    cuda = torch.device("cuda")
    assert is_gpt2_small_model("gpt2")
    assert is_gpt2_small_model("openai-community/gpt2")
    assert resolve_model_dtype("gpt2", "auto", cuda) == torch.float32
    assert resolve_model_dtype("gpt2", "fp32", cuda) == torch.float32
    assert dtype_name(torch.float32) == "float32"


@pytest.mark.parametrize("precision", ["float16", "fp16", "bfloat16", "bf16"])
def test_gpt2_small_rejects_reduced_precision(precision):
    with pytest.raises(ValueError, match="require float32"):
        resolve_model_dtype("gpt2", precision, torch.device("cuda"))


def test_other_models_keep_regular_dtype_resolution():
    assert (
        resolve_model_dtype("some-other-model", "fp16", torch.device("cuda"))
        == torch.float16
    )

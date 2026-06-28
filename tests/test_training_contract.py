from __future__ import annotations

import pytest

from train import _validate_run_name, load_model_module


@pytest.mark.parametrize(
    "run_name",
    ["efficientnet_b0_smoke", "efficientnet-b0-final", "run42"],
)
def test_run_name_accepts_safe_artifact_stems(run_name: str) -> None:
    assert _validate_run_name(run_name) == run_name


@pytest.mark.parametrize("run_name", ["", "../escape", "name with spaces"])
def test_run_name_rejects_unsafe_artifact_stems(run_name: str) -> None:
    with pytest.raises(ValueError):
        _validate_run_name(run_name)


def test_recursive_model_loading_finds_efficientnet_module() -> None:
    name, module = load_model_module("models/EfficientNet-B0/efficientnet_b0.py")

    assert name == "efficientnet_b0"
    assert callable(module.build_model)
    assert callable(module.configure)


@pytest.mark.parametrize(
    ("dropout", "learning_rate"),
    [(-0.1, None), (1.0, None), (None, 0.0), (None, -1e-3)],
)
def test_model_configuration_rejects_invalid_factors(
    dropout: float | None, learning_rate: float | None
) -> None:
    _, module = load_model_module("efficientnet_b0")

    with pytest.raises(ValueError):
        module.configure(dropout=dropout, learning_rate=learning_rate)

"""ONNX detector initialization and contract-validation tests."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.detection.onnx_detector import (
    ModelValidationError,
    OnnxPersonDetector,
    select_providers,
    validate_model_contract,
)


@dataclass
class Metadata:
    name: str
    shape: list[object]
    type: str = "tensor(float)"


def test_invalid_model_path_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="detector model file does not exist"):
        OnnxPersonDetector(tmp_path / "missing.onnx")


def test_dynamic_batch_light_contract_is_accepted() -> None:
    input_meta = Metadata("images", ["batch", 3, 640, 640])
    output_meta = Metadata("output", ["batch", 5, 8400])

    names_and_size = validate_model_contract([input_meta], [output_meta])

    assert names_and_size == ("images", "output", (640, 640))


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "message"),
    [
        ([1, 3, 320, 320], [1, 5, 2100], "input must be"),
        ([2, 3, 640, 640], [2, 5, 8400], "batch dimension"),
        ([1, 3, 640, 640], [1, 300, 6], "output must be"),
        ([1, 3, 640, 640], [1, 84, 8400], "output must be"),
    ],
)
def test_unexpected_model_shapes_are_rejected(
    input_shape: list[object],
    output_shape: list[object],
    message: str,
) -> None:
    with pytest.raises(ModelValidationError, match=message):
        validate_model_contract(
            [Metadata("images", input_shape)],
            [Metadata("output", output_shape)],
        )


def test_provider_selection_adds_cpu_fallback() -> None:
    selected = select_providers(
        ["CUDAExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert selected == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_unavailable_accelerator_falls_back_to_cpu() -> None:
    selected = select_providers(
        ["CUDAExecutionProvider"],
        ["CPUExecutionProvider"],
    )

    assert selected == ("CPUExecutionProvider",)

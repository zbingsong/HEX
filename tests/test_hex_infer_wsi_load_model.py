from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hex import infer_wsi_hex


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state_dict = None
        self.loaded_strict = None
        self.eval_called = False

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[no-untyped-def]
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self


@pytest.mark.parametrize(
    "checkpoint_payload",
    [
        {"encoder.weight": object()},
        {"state_dict": {"encoder.weight": object()}},
        {"model_state_dict": {"encoder.weight": object()}},
    ],
)
def test_load_hex_wsi_model_loads_custom_model_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_payload: dict[str, object],
) -> None:
    checkpoint_path = Path("/tmp/fake-checkpoint.pth")
    fake_model = _FakeModel()
    constructor_calls: list[tuple[int, int]] = []
    load_calls: list[tuple[dict[str, object], bool]] = []

    def fake_custom_model(
        visual_output_dim: int,
        num_outputs: int,
    ) -> _FakeModel:
        constructor_calls.append((visual_output_dim, num_outputs))
        return fake_model

    def fake_torch_load(path: Path, map_location: str) -> dict[str, object]:
        assert path == checkpoint_path
        assert map_location == "cpu"
        return checkpoint_payload

    monkeypatch.setattr(infer_wsi_hex, "CustomModel", fake_custom_model)
    monkeypatch.setattr(infer_wsi_hex.torch, "load", fake_torch_load)

    model = infer_wsi_hex.load_hex_wsi_model(checkpoint_path)

    assert model is fake_model
    assert constructor_calls == [(1024, infer_wsi_hex.HEX_BIOMARKER_COUNT)]
    assert fake_model.loaded_strict is False
    assert fake_model.eval_called is True

    if "state_dict" in checkpoint_payload:
        expected_state_dict = checkpoint_payload["state_dict"]
    elif "model_state_dict" in checkpoint_payload:
        expected_state_dict = checkpoint_payload["model_state_dict"]
    else:
        expected_state_dict = checkpoint_payload

    assert fake_model.loaded_state_dict is expected_state_dict


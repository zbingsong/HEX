from __future__ import annotations

from pathlib import Path

import pytest

from hex import hex_architecture


def test_musk_backbone_checkpoint_path_is_repo_local() -> None:
    assert hex_architecture.MUSK_BACKBONE_CHECKPOINT_PATH == (
        Path(__file__).resolve().parents[1] / "hex" / "model.safetensors"
    )


def test_register_musk_timm_models_imports_musk_modeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_names: list[str] = []

    def fake_import_module(name: str) -> object:
        imported_names.append(name)
        return object()

    monkeypatch.setattr(hex_architecture.importlib, "import_module", fake_import_module)

    hex_architecture._register_musk_timm_models()

    assert imported_names == ["musk.modeling"]


def test_register_musk_timm_models_raises_actionable_error_when_musk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        raise ModuleNotFoundError("No module named 'musk'", name="musk")

    monkeypatch.setattr(hex_architecture.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="Install MUSK from"):
        hex_architecture._register_musk_timm_models()

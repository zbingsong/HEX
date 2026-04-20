from __future__ import annotations

from pathlib import Path

from hex.hex_architecture import MUSK_BACKBONE_CHECKPOINT_PATH


def test_musk_backbone_checkpoint_path_is_repo_local() -> None:
    assert MUSK_BACKBONE_CHECKPOINT_PATH == Path(__file__).resolve().parents[1] / "hex" / "checkpoint.pth"

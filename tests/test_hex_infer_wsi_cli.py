from __future__ import annotations

from pathlib import Path

import pytest

from hex import infer_wsi_hex


def test_hex_infer_wsi_cli_rejects_missing_wsi(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")

    exit_code = infer_wsi_hex.main(
        [
            "--wsi-path",
            str(tmp_path / "missing.svs"),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_hex_infer_wsi_cli_rejects_missing_checkpoint(tmp_path: Path) -> None:
    wsi_path = tmp_path / "slide.svs"
    wsi_path.write_bytes(b"")
    exit_code = infer_wsi_hex.main(
        [
            "--wsi-path",
            str(wsi_path),
            "--checkpoint-path",
            str(tmp_path / "missing-checkpoint.pth"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_hex_infer_wsi_cli_rejects_negative_level(tmp_path: Path) -> None:
    wsi_path = tmp_path / "slide.svs"
    wsi_path.write_bytes(b"")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")

    exit_code = infer_wsi_hex.main(
        [
            "--wsi-path",
            str(wsi_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--level",
            "-1",
        ]
    )

    assert exit_code == 1


def test_hex_infer_wsi_cli_uses_default_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wsi_path = tmp_path / "slide.svs"
    wsi_path.write_bytes(b"")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def fake_run_wsi_hex_inference(**kwargs: object) -> dict[str, Path]:
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "slide_level0_pred.npy"
        skipmask_path = output_dir / "slide_level0_skipmask.npy"
        ome_path = output_dir / "slide.ome.tiff"
        for path in (pred_path, skipmask_path, ome_path):
            path.write_bytes(b"")
        return {
            "prediction_npy": pred_path,
            "skipmask_npy": skipmask_path,
            "metadata_json": output_dir / "slide_level0_metadata.json",
            "ome_tiff": ome_path,
        }

    monkeypatch.setattr(infer_wsi_hex, "run_wsi_hex_inference", fake_run_wsi_hex_inference)

    exit_code = infer_wsi_hex.main(
        [
            "--wsi-path",
            str(wsi_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "wsi_path": wsi_path,
        "checkpoint_path": checkpoint_path,
        "output_dir": tmp_path / "out",
        "level": 0,
        "patch_size": 224,
        "stride": 4,
        "batch_size": 32,
        "num_workers": 0,
        "image_size": 384,
        "device": "auto",
        "white_threshold": 240,
        "min_white_fraction": 0.98,
    }

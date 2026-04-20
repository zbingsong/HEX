from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_hex_infer_wsi_cli_rejects_missing_wsi(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "hex" / "infer_wsi_hex.py"
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--wsi-path",
            str(tmp_path / "missing.svs"),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--level",
            "0",
            "--patch-size",
            "224",
            "--stride",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "WSI path does not exist" in result.stderr


def test_hex_infer_wsi_cli_uses_default_arguments(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "hex" / "infer_wsi_hex.py"
    wsi_path = tmp_path / "slide.svs"
    wsi_path.write_bytes(b"")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--wsi-path",
            str(wsi_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""

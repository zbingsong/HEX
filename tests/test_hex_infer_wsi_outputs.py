from __future__ import annotations

from pathlib import Path

import numpy as np

from hex.infer_wsi_hex import create_disk_backed_level_outputs


def test_create_disk_backed_level_outputs_creates_incrementally_fillable_npy_arrays(
    tmp_path: Path,
) -> None:
    pred_path, skipmask_path, predictions, skip_mask = create_disk_backed_level_outputs(
        output_dir=tmp_path,
        slide_id="slide123",
        level=2,
        rows=3,
        cols=4,
        channels=5,
    )

    assert pred_path == tmp_path / "slide123_level2_pred.npy"
    assert skipmask_path == tmp_path / "slide123_level2_skipmask.npy"
    assert predictions.shape == (3, 4, 5)
    assert skip_mask.shape == (3, 4)
    assert predictions.dtype == np.float16
    assert skip_mask.dtype == np.bool_
    assert str(predictions.filename).endswith(".npy")
    assert str(skip_mask.filename).endswith(".npy")

    predictions[1, 2] = np.arange(5, dtype=np.float32)
    skip_mask[0, 1] = True
    predictions.flush()
    skip_mask.flush()

    reloaded_predictions = np.load(
        tmp_path / "slide123_level2_pred.npy",
        mmap_mode="r",
    )
    reloaded_skip_mask = np.load(
        tmp_path / "slide123_level2_skipmask.npy",
        mmap_mode="r",
    )

    assert np.array_equal(
        reloaded_predictions[1, 2],
        np.arange(5, dtype=np.float32),
    )
    assert bool(reloaded_skip_mask[0, 1]) is True

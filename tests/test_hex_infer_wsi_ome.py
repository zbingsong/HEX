from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from hex.infer_wsi_hex import write_prediction_ome_tiff


def test_write_prediction_ome_tiff_exports_single_level_predictions(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "slide_level0_pred.npy"
    prediction = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    np.save(prediction_path, prediction)
    output_path = tmp_path / "slide.ome.tiff"

    returned_path = write_prediction_ome_tiff(
        level_prediction_paths=[prediction_path],
        output_path=output_path,
        channel_names=["c1", "c2", "c3", "c4"],
    )

    assert returned_path == output_path
    with tifffile.TiffFile(output_path) as tif:
        series = tif.series[0]
        assert series.shape == (4, 2, 3)
        np.testing.assert_array_equal(series.asarray(), np.moveaxis(prediction, -1, 0))

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from hex.infer_wsi_hex import build_prediction_ome_xml, write_prediction_ome_tiff


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
        assert tif.is_ome
        series = tif.series[0]
        assert series.shape == (4, 2, 3)
        assert series.axes == "CYX"
        assert "c4" in tif.ome_metadata
        np.testing.assert_array_equal(series.asarray(), np.moveaxis(prediction, -1, 0))


def test_build_prediction_ome_xml_records_multiple_level_ifd_offsets() -> None:
    ome_xml = build_prediction_ome_xml(
        level_names=["slide_level0_pred", "slide_level1_pred"],
        level_shapes=[(4, 2, 3), (4, 1, 2)],
        channel_names=["c1", "c2", "c3", "c4"],
    )

    assert 'Name="slide_level0_pred"' in ome_xml
    assert 'Name="slide_level1_pred"' in ome_xml
    assert 'IFD="0"' in ome_xml
    assert 'IFD="4"' in ome_xml
    assert 'PlaneCount="4"' in ome_xml

from __future__ import annotations

import pytest

from hex.infer_wsi_hex import PaddedCropBounds, compute_padded_crop_bounds


def test_compute_padded_crop_bounds_returns_interior_crop_without_padding() -> None:
    assert compute_padded_crop_bounds(
        center_x=4,
        center_y=4,
        patch_size=4,
        width=8,
        height=8,
    ) == PaddedCropBounds(2, 2, 6, 6, 0, 0, 0, 0)


def test_compute_padded_crop_bounds_pads_top_left_corner() -> None:
    assert compute_padded_crop_bounds(
        center_x=0,
        center_y=0,
        patch_size=4,
        width=8,
        height=8,
    ) == PaddedCropBounds(0, 0, 2, 2, 2, 2, 0, 0)


def test_compute_padded_crop_bounds_pads_bottom_right_corner() -> None:
    assert compute_padded_crop_bounds(
        center_x=7,
        center_y=7,
        patch_size=4,
        width=8,
        height=8,
    ) == PaddedCropBounds(5, 5, 8, 8, 0, 0, 1, 1)


@pytest.mark.parametrize(
    ("center_x", "center_y", "patch_size", "width", "height"),
    [
        (0, 0, 0, 8, 8),
        (0, 0, -1, 8, 8),
        (0, 0, 4, -1, 8),
        (0, 0, 4, 8, -1),
        (-1, 0, 4, 8, 8),
        (0, -1, 4, 8, 8),
    ],
)
def test_compute_padded_crop_bounds_rejects_invalid_inputs(
    center_x: int,
    center_y: int,
    patch_size: int,
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError):
        compute_padded_crop_bounds(
            center_x=center_x,
            center_y=center_y,
            patch_size=patch_size,
            width=width,
            height=height,
        )

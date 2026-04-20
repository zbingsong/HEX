from __future__ import annotations

import pytest

from hex.infer_wsi_hex import (
    compute_output_grid_shape,
    output_grid_index_to_center_coordinates,
)


@pytest.mark.parametrize(
    ("width", "height", "stride", "expected"),
    [
        (1, 1, 4, (1, 1)),
        (8, 5, 4, (2, 2)),
        (9, 5, 4, (2, 3)),
        (224, 225, 4, (57, 56)),
    ],
)
def test_compute_output_grid_shape_rounds_up(
    width: int, height: int, stride: int, expected: tuple[int, int]
) -> None:
    assert compute_output_grid_shape(width, height, stride) == expected


@pytest.mark.parametrize(
    ("row", "col", "stride", "expected"),
    [
        (0, 0, 4, (0, 0)),
        (1, 2, 4, (8, 4)),
        (3, 5, 4, (20, 12)),
    ],
)
def test_output_grid_index_to_center_coordinates_uses_stride_grid(
    row: int, col: int, stride: int, expected: tuple[int, int]
) -> None:
    assert output_grid_index_to_center_coordinates(row, col, stride) == expected


@pytest.mark.parametrize("stride", [0, -1])
def test_grid_helpers_reject_non_positive_stride(stride: int) -> None:
    with pytest.raises(ValueError):
        compute_output_grid_shape(8, 8, stride)

    with pytest.raises(ValueError):
        output_grid_index_to_center_coordinates(0, 0, stride)


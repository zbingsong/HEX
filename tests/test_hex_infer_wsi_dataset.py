from __future__ import annotations

from dataclasses import dataclass

from hex.infer_wsi_hex import WSISlidingWindowDataset, WSISlidingWindowItem


@dataclass(frozen=True, slots=True)
class FakeSlide:
    level_dimensions: tuple[tuple[int, int], ...]


def test_wsi_sliding_window_dataset_uses_level_dimensions_and_stride_grid() -> None:
    slide = FakeSlide(level_dimensions=((9, 5), (4, 3)))

    dataset = WSISlidingWindowDataset(
        slide=slide,
        level=1,
        patch_size=224,
        stride=4,
        white_threshold=239,
        min_white_fraction=0.97,
    )

    assert dataset.level == 1
    assert dataset.patch_size == 224
    assert dataset.stride == 4
    assert dataset.white_threshold == 239
    assert dataset.min_white_fraction == 0.97
    assert dataset.width == 4
    assert dataset.height == 3
    assert dataset.grid_rows == 1
    assert dataset.grid_cols == 1
    assert len(dataset) == 1

    item = dataset[0]
    assert item == WSISlidingWindowItem(
        index=0,
        row=0,
        col=0,
        center_x=0,
        center_y=0,
    )


def test_wsi_sliding_window_dataset_maps_indices_across_a_dense_grid() -> None:
    slide = FakeSlide(level_dimensions=((9, 5),))

    dataset = WSISlidingWindowDataset(
        slide=slide,
        level=0,
        patch_size=224,
        stride=4,
    )

    assert len(dataset) == 6
    assert dataset.grid_rows == 2
    assert dataset.grid_cols == 3
    assert dataset[0].row == 0
    assert dataset[0].col == 0
    assert dataset[0].center_x == 0
    assert dataset[0].center_y == 0
    assert dataset[5].row == 1
    assert dataset[5].col == 2
    assert dataset[5].center_x == 8
    assert dataset[5].center_y == 4

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from hex.infer_wsi_hex import PaddedCropBounds, WSISlidingWindowDataset


@dataclass(slots=True)
class FakeReadableSlide:
    level_images: tuple[np.ndarray, ...]
    level_downsamples: tuple[float, ...] = (1.0,)
    read_calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = field(
        default_factory=list
    )

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple((image.shape[1], image.shape[0]) for image in self.level_images)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        self.read_calls.append((location, level, size))
        x0, y0 = location
        width, height = size
        image = self.level_images[level]
        downsample = int(round(self.level_downsamples[level]))
        level_x0 = x0 // downsample
        level_y0 = y0 // downsample
        crop = image[level_y0 : level_y0 + height, level_x0 : level_x0 + width]
        return Image.fromarray(crop, mode="RGB")


def test_wsi_sliding_window_dataset_reads_padded_top_left_patch() -> None:
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [255, 255, 255]],
            [[70, 80, 90], [100, 110, 120], [255, 255, 255]],
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    slide = FakeReadableSlide(level_images=(image,))
    dataset = WSISlidingWindowDataset(
        slide=slide,
        level=0,
        patch_size=4,
        stride=4,
        white_threshold=240,
        min_white_fraction=0.9,
    )

    item = dataset[0]

    assert item.crop_bounds == PaddedCropBounds(0, 0, 2, 2, 2, 2, 0, 0)
    assert item.patch_rgb is not None
    assert item.patch_rgb.shape == (4, 4, 3)
    assert np.all(item.patch_rgb[:2, :, :] == 255)
    assert np.all(item.patch_rgb[:, :2, :] == 255)
    np.testing.assert_array_equal(item.patch_rgb[2:, 2:, :], image[:2, :2, :])
    assert item.is_near_white is False


def test_wsi_sliding_window_dataset_marks_near_white_patch_as_skipped() -> None:
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    slide = FakeReadableSlide(level_images=(image,))
    dataset = WSISlidingWindowDataset(
        slide=slide,
        level=0,
        patch_size=4,
        stride=4,
    )

    item = dataset[0]

    assert item.patch_rgb is not None
    assert item.is_near_white is True


def test_wsi_sliding_window_dataset_scales_level_coordinates_for_read_region() -> None:
    level0_image = np.zeros((8, 8, 3), dtype=np.uint8)
    level1_image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    slide = FakeReadableSlide(
        level_images=(level0_image, level1_image),
        level_downsamples=(1.0, 2.0),
    )
    dataset = WSISlidingWindowDataset(
        slide=slide,
        level=1,
        patch_size=2,
        stride=2,
        white_threshold=300 - 45,  # keep the test explicit without default reliance
        min_white_fraction=1.0,
    )

    item = dataset[3]

    assert slide.read_calls == [((2, 2), 1, (2, 2))]
    assert item.crop_bounds == PaddedCropBounds(1, 1, 3, 3, 0, 0, 0, 0)
    assert item.patch_rgb is not None
    np.testing.assert_array_equal(item.patch_rgb, level1_image[1:3, 1:3, :])

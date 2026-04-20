"""CLI skeleton for future HEX WSI inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class PaddedCropBounds:
    """Bounds and padding needed to realize a centered patch request."""

    read_x0: int
    read_y0: int
    read_x1: int
    read_y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


class SupportsLevelDimensions(Protocol):
    """Minimal slide protocol for level-dimension lookups."""

    level_dimensions: Sequence[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class WSISlidingWindowItem:
    """Index metadata for one output-grid location."""

    index: int
    row: int
    col: int
    center_x: int
    center_y: int


class WSISlidingWindowDataset(Dataset):
    """Dense sliding-window index shell for one WSI level.

    The dataset only computes output-grid indexing and level-space center
    coordinates. Patch reading and inference are deferred to later tasks.
    """

    def __init__(
        self,
        slide: SupportsLevelDimensions,
        level: int,
        patch_size: int = 224,
        stride: int = 4,
        white_threshold: int = 240,
        min_white_fraction: float = 0.98,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if white_threshold < 0 or white_threshold > 255:
            raise ValueError("white_threshold must be between 0 and 255")
        if min_white_fraction < 0.0 or min_white_fraction > 1.0:
            raise ValueError("min_white_fraction must be between 0 and 1")

        try:
            width, height = slide.level_dimensions[level]
        except IndexError as exc:
            raise IndexError(
                f"level {level} is out of range for slide.level_dimensions"
            ) from exc

        self.slide = slide
        self.level = level
        self.patch_size = patch_size
        self.stride = stride
        self.white_threshold = white_threshold
        self.min_white_fraction = min_white_fraction
        self.width = int(width)
        self.height = int(height)
        self.grid_rows, self.grid_cols = compute_output_grid_shape(
            self.width, self.height, self.stride
        )

    def __len__(self) -> int:
        return self.grid_rows * self.grid_cols

    def __getitem__(self, index: int) -> WSISlidingWindowItem:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        row = index // self.grid_cols
        col = index % self.grid_cols
        center_x, center_y = output_grid_index_to_center_coordinates(
            row, col, self.stride
        )
        return WSISlidingWindowItem(
            index=index,
            row=row,
            col=col,
            center_x=center_x,
            center_y=center_y,
        )


def is_near_white_patch(
    patch: np.ndarray,
    threshold: int = 240,
    min_white_fraction: float = 0.98,
) -> bool:
    """Return ``True`` when an RGB patch is skippable as near-white background.

    A patch qualifies when at least ``min_white_fraction`` of its pixels have
    all three RGB channels strictly greater than ``threshold``.
    """

    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be between 0 and 255")
    if min_white_fraction < 0.0 or min_white_fraction > 1.0:
        raise ValueError("min_white_fraction must be between 0 and 1")

    rgb_patch = np.asarray(patch)
    if rgb_patch.ndim != 3 or rgb_patch.shape[-1] != 3:
        raise ValueError("patch must have shape (height, width, 3)")
    if rgb_patch.size == 0:
        return False

    white_pixels = np.all(rgb_patch > threshold, axis=-1)
    white_fraction = np.count_nonzero(white_pixels) / white_pixels.size
    return white_fraction >= min_white_fraction


def compute_output_grid_shape(width: int, height: int, stride: int) -> tuple[int, int]:
    """Return the dense output-grid shape as ``(rows, cols)``.

    The grid is defined in stride space, so each dimension rounds up to cover
    the full level extent.
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")

    rows = (height + stride - 1) // stride
    cols = (width + stride - 1) // stride
    return rows, cols


def output_grid_index_to_center_coordinates(
    row: int, col: int, stride: int
) -> tuple[int, int]:
    """Map an output-grid index to the centered level-space coordinates.

    Returns ``(center_x, center_y)`` so the result can be fed directly into
    level-space patch extraction.
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")

    return col * stride, row * stride


def compute_padded_crop_bounds(
    center_x: int,
    center_y: int,
    patch_size: int,
    width: int,
    height: int,
) -> PaddedCropBounds:
    """Return the in-bounds crop and padding for a centered patch request.

    The returned object captures the readable region inside the WSI together
    with the padding required on each side to reconstruct the full centered
    patch.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")
    if center_x < 0 or center_y < 0:
        raise ValueError("center_x and center_y must be non-negative")

    half_size = patch_size // 2
    requested_x0 = center_x - half_size
    requested_y0 = center_y - half_size
    requested_x1 = requested_x0 + patch_size
    requested_y1 = requested_y0 + patch_size

    read_x0 = max(requested_x0, 0)
    read_y0 = max(requested_y0, 0)
    read_x1 = min(requested_x1, width)
    read_y1 = min(requested_y1, height)

    pad_left = read_x0 - requested_x0
    pad_top = read_y0 - requested_y0
    pad_right = requested_x1 - read_x1
    pad_bottom = requested_y1 - read_y1

    return PaddedCropBounds(
        read_x0=read_x0,
        read_y0=read_y0,
        read_x1=read_x1,
        read_y1=read_y1,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for HEX WSI inference."""

    parser = argparse.ArgumentParser(description="Run HEX inference on a WSI.")
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and validate the WSI path."""

    args = build_parser().parse_args(argv)

    if not args.wsi_path.exists():
        print("WSI path does not exist", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

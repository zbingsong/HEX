"""CLI skeleton for future HEX WSI inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image
import torch
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from torch.utils.data import Dataset
from numpy.lib.format import open_memmap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hex.hex_architecture import CustomModel
else:
    from .hex_architecture import CustomModel


HEX_BIOMARKER_NAMES: dict[int, str] = {
    1: "DAPI",
    2: "CD8",
    3: "Pan-Cytokeratin",
    4: "CD3e",
    5: "CD163",
    6: "CD20",
    7: "CD4",
    8: "FAP",
    9: "CD138",
    10: "CD11c",
    11: "CD66b",
    12: "aSMA",
    13: "CD68",
    14: "Ki67",
    15: "CD31",
    16: "Collagen IV",
    17: "Granzyme B",
    18: "MMP9",
    19: "PD-1",
    20: "CD44",
    21: "PD-L1",
    22: "E-cadherin",
    23: "LAG3",
    24: "Mac2/Galectin-3",
    25: "FOXP3",
    26: "CD14",
    27: "EpCAM",
    28: "CD21",
    29: "CD45",
    30: "MPO",
    31: "TCF-1",
    32: "ICOS",
    33: "Bcl-2",
    34: "HLA-E",
    35: "CD45RO",
    36: "VISTA",
    37: "HIF1A",
    38: "CD39",
    39: "CD40",
    40: "HLA-DR",
}

HEX_BIOMARKER_COUNT = len(HEX_BIOMARKER_NAMES)
HEX_LABEL_COLUMNS: tuple[str, ...] = tuple(
    f"mean_intensity_channel{i}" for i in range(1, HEX_BIOMARKER_COUNT + 1)
)
HEX_MODEL_VISUAL_OUTPUT_DIM = 1024
HEX_MODEL_NUM_OUTPUTS = HEX_BIOMARKER_COUNT


def build_hex_eval_transform(image_size: int = 384) -> transforms.Compose:
    """Return the HEX eval transform used for codex patch inference."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD,
            ),
        ]
    )


def _unwrap_hex_checkpoint_state_dict(checkpoint: Any) -> Any:
    """Return the state dict from a supported HEX checkpoint wrapper."""

    if not isinstance(checkpoint, dict):
        return checkpoint

    for wrapper_key in ("state_dict", "model_state_dict", "model", "module"):
        wrapped_state_dict = checkpoint.get(wrapper_key)
        if isinstance(wrapped_state_dict, dict):
            return wrapped_state_dict

    return checkpoint


def load_hex_wsi_model(checkpoint_path: Path) -> CustomModel:
    """Load the HEX WSI model on CPU and switch it to eval mode.

    The helper mirrors the repo's existing `CustomModel` shape assumptions:
    `visual_output_dim=1024` and `num_outputs=40`.
    """

    model = CustomModel(
        visual_output_dim=HEX_MODEL_VISUAL_OUTPUT_DIM,
        num_outputs=HEX_MODEL_NUM_OUTPUTS,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_hex_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def create_disk_backed_level_outputs(
    output_dir: Path,
    slide_id: str,
    level: int,
    rows: int,
    cols: int,
    channels: int = HEX_BIOMARKER_COUNT,
    prediction_dtype: np.dtype[Any] | type[np.floating] = np.float16,
    skip_mask_dtype: np.dtype[Any] | type[np.bool_] = np.bool_,
) -> tuple[Path, Path, np.memmap, np.memmap]:
    """Create `.npy`-backed arrays for one WSI level's outputs.

    The prediction raster is shaped ``(rows, cols, channels)`` and the skip
    mask is shaped ``(rows, cols)``. Both arrays are created in ``w+`` mode so
    they can be filled incrementally and flushed to disk later.
    """

    if not slide_id:
        raise ValueError("slide_id must be non-empty")
    if level < 0:
        raise ValueError("level must be non-negative")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{slide_id}_level{level}_pred.npy"
    skipmask_path = output_dir / f"{slide_id}_level{level}_skipmask.npy"

    predictions = open_memmap(
        pred_path,
        mode="w+",
        dtype=prediction_dtype,
        shape=(rows, cols, channels),
    )
    skip_mask = open_memmap(
        skipmask_path,
        mode="w+",
        dtype=skip_mask_dtype,
        shape=(rows, cols),
    )
    return pred_path, skipmask_path, predictions, skip_mask


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


class SupportsWSIReadRegion(SupportsLevelDimensions, Protocol):
    """Minimal slide protocol for region reads across pyramid levels."""

    level_downsamples: Sequence[float]

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        """Read one region from the slide pyramid."""


@dataclass(frozen=True, slots=True)
class WSISlidingWindowItem:
    """One centered patch request and its extracted payload, if available."""

    index: int
    row: int
    col: int
    center_x: int
    center_y: int
    patch_rgb: np.ndarray | None = None
    crop_bounds: PaddedCropBounds | None = None
    is_near_white: bool = False


class WSISlidingWindowDataset(Dataset):
    """Dense sliding-window dataset for one WSI level.

    Each item is aligned to one output-grid location in stride space. When the
    provided slide implements ``read_region``, the dataset also returns the
    centered RGB patch, symmetric padding metadata, and a near-white skip flag.
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
        item = WSISlidingWindowItem(
            index=index,
            row=row,
            col=col,
            center_x=center_x,
            center_y=center_y,
        )
        if not hasattr(self.slide, "read_region"):
            return item

        patch_rgb, crop_bounds = read_centered_padded_patch(
            slide=self.slide,
            level=self.level,
            center_x=center_x,
            center_y=center_y,
            patch_size=self.patch_size,
            width=self.width,
            height=self.height,
        )
        return WSISlidingWindowItem(
            index=item.index,
            row=item.row,
            col=item.col,
            center_x=item.center_x,
            center_y=item.center_y,
            patch_rgb=patch_rgb,
            crop_bounds=crop_bounds,
            is_near_white=is_near_white_patch(
                patch_rgb,
                threshold=self.white_threshold,
                min_white_fraction=self.min_white_fraction,
            ),
        )


def _get_level_downsample(slide: SupportsLevelDimensions, level: int) -> float:
    """Return the level downsample, defaulting to ``1.0`` for test doubles."""

    level_downsamples = getattr(slide, "level_downsamples", None)
    if level_downsamples is None:
        return 1.0

    try:
        downsample = float(level_downsamples[level])
    except IndexError as exc:
        raise IndexError(
            f"level {level} is out of range for slide.level_downsamples"
        ) from exc

    if downsample <= 0:
        raise ValueError("slide level downsample must be positive")
    return downsample


def _region_to_rgb_array(region: Image.Image | np.ndarray) -> np.ndarray:
    """Convert one slide region into a ``uint8`` RGB array."""

    if isinstance(region, np.ndarray):
        region_array = np.asarray(region)
        if region_array.ndim != 3 or region_array.shape[-1] not in (3, 4):
            raise ValueError("region array must have shape (height, width, 3|4)")
        if region_array.dtype != np.uint8:
            region_array = region_array.astype(np.uint8, copy=False)
        if region_array.shape[-1] == 4:
            region_array = region_array[..., :3]
        return region_array

    return np.asarray(region.convert("RGB"), dtype=np.uint8)


def read_centered_padded_patch(
    slide: SupportsLevelDimensions,
    level: int,
    center_x: int,
    center_y: int,
    patch_size: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, PaddedCropBounds]:
    """Read one centered RGB patch with symmetric white padding.

    Returns a ``uint8`` array with shape ``(patch_size, patch_size, 3)`` in
    ``RGB`` order plus the in-bounds crop metadata used to place it.
    """

    if not hasattr(slide, "read_region"):
        raise TypeError("slide must provide read_region() for patch extraction")

    crop_bounds = compute_padded_crop_bounds(
        center_x=center_x,
        center_y=center_y,
        patch_size=patch_size,
        width=width,
        height=height,
    )
    read_width = crop_bounds.read_x1 - crop_bounds.read_x0
    read_height = crop_bounds.read_y1 - crop_bounds.read_y0

    # The patch canvas is always full-sized in level space: (H, W, C) uint8 RGB.
    patch_rgb = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    if read_width == 0 or read_height == 0:
        return patch_rgb, crop_bounds

    level_downsample = _get_level_downsample(slide, level)
    level0_x = int(round(crop_bounds.read_x0 * level_downsample))
    level0_y = int(round(crop_bounds.read_y0 * level_downsample))
    region = slide.read_region(
        (level0_x, level0_y),
        level,
        (read_width, read_height),
    )
    region_rgb = _region_to_rgb_array(region)
    if region_rgb.shape != (read_height, read_width, 3):
        raise ValueError(
            "slide region shape does not match requested crop size"
        )

    insert_y0 = crop_bounds.pad_top
    insert_x0 = crop_bounds.pad_left
    insert_y1 = insert_y0 + read_height
    insert_x1 = insert_x0 + read_width
    patch_rgb[insert_y0:insert_y1, insert_x0:insert_x1] = region_rgb
    return patch_rgb, crop_bounds


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

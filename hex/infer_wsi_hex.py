"""CLI skeleton for future HEX WSI inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

"""CLI skeleton for future HEX WSI inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

# HEX WSI Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone HEX inference script that reads one `.svs` WSI, runs centered sliding-window inference on a chosen level, saves a dense stride-space `.npy` prediction raster plus skip mask, and exports an OME-TIFF.

**Architecture:** Add a new `hex/` script that reuses the existing HEX model and normalization path, but replaces pre-extracted PNG patches with an OpenSlide-backed sliding-window dataset. Write predictions into disk-backed arrays to keep memory bounded, then convert the saved per-level arrays into an OME-TIFF export artifact.

**Tech Stack:** Python, PyTorch, OpenSlide, NumPy, tifffile, torchvision, existing HEX model code in `hex/`

---

### Task 1: Create shared WSI inference spec and CLI skeleton

**Files:**
- Create: `hex/infer_wsi_hex.py`
- Test: `tests/test_hex_infer_wsi_cli.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
import subprocess
import sys


def test_cli_rejects_missing_wsi(tmp_path: Path) -> None:
    script = Path("hex/infer_wsi_hex.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--wsi-path",
            str(tmp_path / "missing.svs"),
            "--checkpoint-path",
            "hex/checkpoint.pth",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "WSI path does not exist" in result.stderr
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_cli.py::test_cli_rejects_missing_wsi -v`
Expected: FAIL because `hex/infer_wsi_hex.py` does not exist yet

**Step 3: Write minimal implementation**

```python
import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not Path(args.wsi_path).exists():
        print("WSI path does not exist", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_cli.py::test_cli_rejects_missing_wsi -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_cli.py hex/infer_wsi_hex.py
git commit -m "feat: add HEX WSI inference CLI skeleton"
```

### Task 2: Add coordinate-grid utilities for centered stride-space sampling

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_grid.py`

**Step 1: Write the failing test**

```python
from hex.infer_wsi_hex import compute_output_grid_shape, output_index_to_center


def test_grid_shape_uses_stride_ceiling() -> None:
    assert compute_output_grid_shape(width=2000, height=2000, stride=4) == (500, 500)


def test_first_center_is_top_left_corner() -> None:
    assert output_index_to_center(row=0, col=0, stride=4) == (0, 0)


def test_center_scales_by_stride() -> None:
    assert output_index_to_center(row=3, col=5, stride=4) == (20, 12)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_grid.py -v`
Expected: FAIL with missing functions/import errors

**Step 3: Write minimal implementation**

```python
import math


def compute_output_grid_shape(width: int, height: int, stride: int) -> tuple[int, int]:
    return math.ceil(height / stride), math.ceil(width / stride)


def output_index_to_center(row: int, col: int, stride: int) -> tuple[int, int]:
    return col * stride, row * stride
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_grid.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_grid.py hex/infer_wsi_hex.py
git commit -m "feat: add centered stride-space grid helpers"
```

### Task 3: Add padded patch extraction logic

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_padding.py`

**Step 1: Write the failing test**

```python
import numpy as np

from hex.infer_wsi_hex import compute_padded_crop_bounds


def test_top_left_center_requires_leading_padding() -> None:
    bounds = compute_padded_crop_bounds(center_x=0, center_y=0, width=2000, height=2000, patch_size=224)
    assert bounds.read_x0 == 0
    assert bounds.read_y0 == 0
    assert bounds.pad_left == 112
    assert bounds.pad_top == 112


def test_bottom_right_center_requires_trailing_padding() -> None:
    bounds = compute_padded_crop_bounds(center_x=1999, center_y=1999, width=2000, height=2000, patch_size=224)
    assert bounds.pad_right > 0
    assert bounds.pad_bottom > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_padding.py -v`
Expected: FAIL with missing helper/type errors

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CropBounds:
    read_x0: int
    read_y0: int
    read_x1: int
    read_y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def compute_padded_crop_bounds(
    center_x: int, center_y: int, width: int, height: int, patch_size: int
) -> CropBounds:
    half = patch_size // 2
    x0 = center_x - half
    y0 = center_y - half
    x1 = center_x + half
    y1 = center_y + half
    read_x0 = max(0, x0)
    read_y0 = max(0, y0)
    read_x1 = min(width, x1)
    read_y1 = min(height, y1)
    return CropBounds(
        read_x0=read_x0,
        read_y0=read_y0,
        read_x1=read_x1,
        read_y1=read_y1,
        pad_left=max(0, -x0),
        pad_top=max(0, -y0),
        pad_right=max(0, x1 - width),
        pad_bottom=max(0, y1 - height),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_padding.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_padding.py hex/infer_wsi_hex.py
git commit -m "feat: add symmetric padded crop helpers"
```

### Task 4: Add near-white patch filtering

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_background.py`

**Step 1: Write the failing test**

```python
import numpy as np

from hex.infer_wsi_hex import is_near_white_patch


def test_near_white_patch_is_skipped() -> None:
    patch = np.full((224, 224, 3), 255, dtype=np.uint8)
    assert is_near_white_patch(patch, white_threshold=240, min_fraction=0.98)


def test_nonwhite_patch_is_not_skipped() -> None:
    patch = np.full((224, 224, 3), 255, dtype=np.uint8)
    patch[50:100, 50:100, :] = 100
    assert not is_near_white_patch(patch, white_threshold=240, min_fraction=0.98)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_background.py -v`
Expected: FAIL with missing helper

**Step 3: Write minimal implementation**

```python
import numpy as np


def is_near_white_patch(
    patch: np.ndarray, white_threshold: int, min_fraction: float
) -> bool:
    white = np.all(patch >= white_threshold, axis=2)
    return float(white.mean()) >= min_fraction
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_background.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_background.py hex/infer_wsi_hex.py
git commit -m "feat: add near-white patch filter"
```

### Task 5: Build the OpenSlide-backed dataset

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_dataset.py`

**Step 1: Write the failing test**

```python
import numpy as np

from hex.infer_wsi_hex import WSISlidingWindowDataset


class FakeSlide:
    level_dimensions = [(16, 16)]

    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., :3] = 255
        return arr


def test_dataset_len_matches_output_grid() -> None:
    ds = WSISlidingWindowDataset(
        slide=FakeSlide(),
        level=0,
        patch_size=224,
        stride=4,
        white_threshold=240,
        min_white_fraction=0.98,
    )
    assert len(ds) == 16
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_dataset.py -v`
Expected: FAIL with missing dataset class

**Step 3: Write minimal implementation**

```python
from torch.utils.data import Dataset


class WSISlidingWindowDataset(Dataset):
    def __init__(self, slide, level, patch_size, stride, white_threshold, min_white_fraction):
        self.slide = slide
        self.level = level
        self.patch_size = patch_size
        self.stride = stride
        self.white_threshold = white_threshold
        self.min_white_fraction = min_white_fraction
        width, height = slide.level_dimensions[level]
        self.rows, self.cols = compute_output_grid_shape(width, height, stride)

    def __len__(self):
        return self.rows * self.cols
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_dataset.py hex/infer_wsi_hex.py
git commit -m "feat: add WSI sliding window dataset shell"
```

### Task 6: Reuse HEX model loading and normalization for WSI inference

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Reference: `hex/test_codex_lung_marker.py`
- Test: `tests/test_hex_infer_wsi_model.py`

**Step 1: Write the failing test**

```python
from hex.infer_wsi_hex import build_eval_transform, biomarker_names


def test_eval_transform_exists() -> None:
    transform = build_eval_transform()
    assert transform is not None


def test_biomarker_count_is_40() -> None:
    assert len(biomarker_names) == 40
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_model.py -v`
Expected: FAIL with missing symbols

**Step 3: Write minimal implementation**

```python
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from hex_architecture import CustomModel


biomarker_names = { ...same 40-marker mapping... }


def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_model.py hex/infer_wsi_hex.py
git commit -m "feat: reuse HEX model metadata and eval transform"
```

### Task 7: Implement batched inference with disk-backed prediction and skip-mask outputs

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_outputs.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
import numpy as np

from hex.infer_wsi_hex import create_output_memmaps


def test_output_memmaps_have_expected_shape(tmp_path: Path) -> None:
    pred_path, mask_path, pred_mm, mask_mm = create_output_memmaps(
        output_dir=tmp_path,
        slide_id="case1",
        level=0,
        rows=5,
        cols=7,
        channels=40,
    )
    assert pred_mm.shape == (5, 7, 40)
    assert mask_mm.shape == (5, 7)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_outputs.py -v`
Expected: FAIL with missing helper

**Step 3: Write minimal implementation**

```python
from numpy.lib.format import open_memmap
from pathlib import Path


def create_output_memmaps(output_dir: Path, slide_id: str, level: int, rows: int, cols: int, channels: int):
    pred_path = output_dir / f"{slide_id}_level{level}_pred.npy"
    mask_path = output_dir / f"{slide_id}_level{level}_skipmask.npy"
    pred_mm = open_memmap(pred_path, mode="w+", dtype=np.float16, shape=(rows, cols, channels))
    mask_mm = open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(rows, cols))
    return pred_path, mask_path, pred_mm, mask_mm
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_outputs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_outputs.py hex/infer_wsi_hex.py
git commit -m "feat: add disk-backed output writers"
```

### Task 8: Add a smoke-mode end-to-end inference path

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_smoke.py`

**Step 1: Write the failing test**

```python
def test_smoke_mode_limits_processed_grid(fake_slide, fake_model, tmp_path):
    # Build a tiny end-to-end inference run with max_rows/max_cols limits.
    # Assert that output arrays are created and only the limited region is touched.
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_smoke.py -v`
Expected: FAIL because the smoke path does not exist

**Step 3: Write minimal implementation**

```python
# Add optional args:
# --max-rows
# --max-cols
# During inference, slice the output-grid iteration to those limits.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_smoke.py hex/infer_wsi_hex.py
git commit -m "feat: add smoke mode for HEX WSI inference"
```

### Task 9: Export predictions to OME-TIFF

**Files:**
- Modify: `hex/infer_wsi_hex.py`
- Create: `tests/test_hex_infer_wsi_ometiff.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
import numpy as np

from hex.infer_wsi_hex import export_levels_to_ome_tiff


def test_ome_tiff_is_written(tmp_path: Path) -> None:
    level_arrays = [np.zeros((5, 7, 40), dtype=np.float16)]
    output_path = tmp_path / "case1.ome.tiff"
    export_levels_to_ome_tiff(
        level_arrays=level_arrays,
        output_path=output_path,
        biomarker_names=[f"m{i}" for i in range(40)],
    )
    assert output_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hex_infer_wsi_ometiff.py -v`
Expected: FAIL with missing exporter

**Step 3: Write minimal implementation**

```python
import tifffile


def export_levels_to_ome_tiff(level_arrays, output_path, biomarker_names):
    arr = np.moveaxis(level_arrays[0], -1, 0)  # C, Y, X
    tifffile.imwrite(
        output_path,
        arr,
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": list(biomarker_names)}},
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hex_infer_wsi_ometiff.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hex_infer_wsi_ometiff.py hex/infer_wsi_hex.py
git commit -m "feat: add OME-TIFF export for HEX WSI inference"
```

### Task 10: Add user-facing documentation for running the new script

**Files:**
- Modify: `README.md`
- Test: manual command example in docs

**Step 1: Write the failing doc check**

```text
Confirm README lacks any WSI inference command for HEX.
```

**Step 2: Run check to verify the gap**

Run: `rg -n "infer_wsi_hex|WSI inference|virtual proteomics" README.md`
Expected: no matching command for the new script

**Step 3: Write minimal implementation**

```markdown
## HEX WSI inference

Run:
`cd hex && python infer_wsi_hex.py --wsi-path /path/to/slide.svs --checkpoint-path checkpoint.pth --output-dir /path/to/out --level 0 --stride 4`
```

**Step 4: Run check to verify it passes**

Run: `rg -n "infer_wsi_hex" README.md`
Expected: one matching section with an example command

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document HEX WSI inference script"
```

### Task 11: Run verification commands for the completed feature

**Files:**
- Verify: `hex/infer_wsi_hex.py`
- Verify: `tests/test_hex_infer_wsi_*.py`
- Verify: `README.md`

**Step 1: Run the targeted test suite**

Run: `pytest tests/test_hex_infer_wsi_cli.py tests/test_hex_infer_wsi_grid.py tests/test_hex_infer_wsi_padding.py tests/test_hex_infer_wsi_background.py tests/test_hex_infer_wsi_dataset.py tests/test_hex_infer_wsi_model.py tests/test_hex_infer_wsi_outputs.py tests/test_hex_infer_wsi_smoke.py tests/test_hex_infer_wsi_ometiff.py -v`
Expected: all tests PASS

**Step 2: Run a smoke inference command**

Run: `cd hex && python infer_wsi_hex.py --wsi-path /path/to/slide.svs --checkpoint-path checkpoint.pth --output-dir /tmp/hex_wsi_smoke --level 0 --stride 4 --max-rows 32 --max-cols 32`
Expected: `.npy`, skip-mask, and `.ome.tiff` outputs are created without crashing

**Step 3: Inspect output metadata**

Run: `python - <<'PY'\nimport numpy as np\narr = np.load('/tmp/hex_wsi_smoke/<slide>_level0_pred.npy', mmap_mode='r')\nmask = np.load('/tmp/hex_wsi_smoke/<slide>_level0_skipmask.npy', mmap_mode='r')\nprint(arr.shape)\nprint(mask.shape)\nPY`
Expected: prediction shape `(rows, cols, 40)` and mask shape `(rows, cols)`

**Step 4: Commit**

```bash
git add hex/infer_wsi_hex.py tests/test_hex_infer_wsi_*.py README.md
git commit -m "feat: add HEX WSI virtual proteomics inference pipeline"
```

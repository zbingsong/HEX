from __future__ import annotations

import numpy as np

from hex.infer_wsi_hex import is_near_white_patch


def test_is_near_white_patch_returns_true_for_fully_white_rgb_patch() -> None:
    patch = np.full((4, 4, 3), 255, dtype=np.uint8)

    assert is_near_white_patch(patch)


def test_is_near_white_patch_returns_false_for_patch_with_tissue() -> None:
    patch = np.full((4, 4, 3), 255, dtype=np.uint8)
    patch[1, 1] = np.array([180, 120, 90], dtype=np.uint8)

    assert not is_near_white_patch(patch)

from __future__ import annotations

import pytest
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

from hex.infer_wsi_hex import (
    HEX_BIOMARKER_COUNT,
    HEX_BIOMARKER_NAMES,
    HEX_LABEL_COLUMNS,
    build_hex_eval_transform,
)


def test_hex_biomarker_metadata_matches_codex_lung_marker_order() -> None:
    assert HEX_BIOMARKER_COUNT == 40
    assert HEX_BIOMARKER_NAMES == {
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
    assert HEX_LABEL_COLUMNS == tuple(
        f"mean_intensity_channel{i}" for i in range(1, 41)
    )


def test_build_hex_eval_transform_matches_existing_hex_normalization() -> None:
    transform = build_hex_eval_transform()

    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 3

    resize, to_tensor, normalize = transform.transforms
    assert isinstance(resize, transforms.Resize)
    assert resize.size == (384, 384)
    assert isinstance(to_tensor, transforms.ToTensor)
    assert isinstance(normalize, transforms.Normalize)
    assert tuple(normalize.mean) == tuple(IMAGENET_INCEPTION_MEAN)
    assert tuple(normalize.std) == tuple(IMAGENET_INCEPTION_STD)


def test_build_hex_eval_transform_supports_custom_image_size() -> None:
    transform = build_hex_eval_transform(image_size=512)

    resize = transform.transforms[0]
    assert isinstance(resize, transforms.Resize)
    assert resize.size == (512, 512)


@pytest.mark.parametrize("image_size", [0, -1])
def test_build_hex_eval_transform_rejects_non_positive_image_size(
    image_size: int,
) -> None:
    with pytest.raises(ValueError):
        build_hex_eval_transform(image_size=image_size)

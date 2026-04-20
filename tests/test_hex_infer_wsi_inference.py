from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from hex.infer_wsi_hex import (
    HEX_BIOMARKER_COUNT,
    WSISlidingWindowItem,
    create_disk_backed_level_outputs,
    run_hex_level_inference,
)


@dataclass(slots=True)
class StaticWSIDataset:
    items: list[WSISlidingWindowItem]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> WSISlidingWindowItem:
        return self.items[index]


class FakeHexModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = x[:, 0, 0, 0].to(dtype=torch.float32).unsqueeze(1)
        preds = base.repeat(1, HEX_BIOMARKER_COUNT)
        features = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        return preds, features


def identity_transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def test_run_hex_level_inference_fills_predictions_and_skip_mask(
    tmp_path: Path,
) -> None:
    dataset = StaticWSIDataset(
        items=[
            WSISlidingWindowItem(
                index=0,
                row=0,
                col=0,
                center_x=0,
                center_y=0,
                patch_rgb=np.full((2, 2, 3), 10, dtype=np.uint8),
                is_near_white=False,
            ),
            WSISlidingWindowItem(
                index=1,
                row=0,
                col=1,
                center_x=2,
                center_y=0,
                patch_rgb=np.full((2, 2, 3), 255, dtype=np.uint8),
                is_near_white=True,
            ),
            WSISlidingWindowItem(
                index=2,
                row=1,
                col=0,
                center_x=0,
                center_y=2,
                patch_rgb=np.full((2, 2, 3), 25, dtype=np.uint8),
                is_near_white=False,
            ),
        ]
    )
    _, _, predictions, skip_mask = create_disk_backed_level_outputs(
        output_dir=tmp_path,
        slide_id="slide123",
        level=0,
        rows=2,
        cols=2,
    )

    run_hex_level_inference(
        model=FakeHexModel(),
        dataset=dataset,
        transform=identity_transform,
        predictions=predictions,
        skip_mask=skip_mask,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    reloaded_predictions = np.load(tmp_path / "slide123_level0_pred.npy")
    reloaded_skip_mask = np.load(tmp_path / "slide123_level0_skipmask.npy")

    assert np.all(reloaded_predictions[0, 0] == np.float16(10))
    assert np.all(reloaded_predictions[1, 0] == np.float16(25))
    assert np.all(reloaded_predictions[0, 1] == np.float16(0))
    assert bool(reloaded_skip_mask[0, 1]) is True
    assert bool(reloaded_skip_mask[0, 0]) is False
    assert bool(reloaded_skip_mask[1, 0]) is False

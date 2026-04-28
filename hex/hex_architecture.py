import importlib
from pathlib import Path
import torch
import torch.nn as nn
from timm import create_model
from . import musk_utils

MUSK_BACKBONE_CHECKPOINT_PATH = Path(__file__).with_name("model.safetensors")


def _register_musk_timm_models() -> None:
    """Import MUSK's modeling module so its timm models are registered."""

    try:
        importlib.import_module("musk.modeling")
    except ModuleNotFoundError as exc:
        if exc.name not in {"musk", "musk.modeling"}:
            raise
        raise ModuleNotFoundError(
            "The MUSK package is required to register timm model "
            "'musk_large_patch16_384'. Install MUSK from "
            "https://github.com/lilab-stanford/MUSK and ensure it is "
            "importable in the active environment."
        ) from exc


class CustomModel(nn.Module):
    def __init__(self, visual_output_dim: int, num_outputs: int) -> None:
        super(CustomModel, self).__init__()
        model_config = "musk_large_patch16_384"
        _register_musk_timm_models()
        if not MUSK_BACKBONE_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"MUSK backbone weights not found at {MUSK_BACKBONE_CHECKPOINT_PATH}. "
                "Place model.safetensors in hex/ before running inference."
            )
        print("Creating model %s" % model_config)
        model_musk = create_model(model_config, vocab_size=64010)
        print("Loaded model checkpoint from %s" % MUSK_BACKBONE_CHECKPOINT_PATH)
        musk_utils.load_model_and_may_interpolate(
            str(MUSK_BACKBONE_CHECKPOINT_PATH),
            model_musk,
            'model|module',
            '',
        )
        self.visual = model_musk
        self.regression_head = nn.Sequential(
            nn.Linear(visual_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.regression_head1 = nn.Sequential(
            nn.Linear(128, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.visual(
            image=x,
            with_head=False,
            out_norm=False
        )[0]
        features = self.regression_head(x)
        preds = self.regression_head1(features)
        return preds, features

# CODEX_REPO_OVERVIEW

## 1. Summary

Observed: This repository is a small research codebase for two related pathology models. `hex/` predicts 40 spatial proteomics marker intensities from H&E image patches, while `mica/` consumes slide-level histology features plus HEX-derived CODEX-like features to predict survival-type clinical outcomes with a co-attention model.

Observed: The codebase is structurally simple but not packaged. Most behavior lives in standalone Python scripts with local helper modules, and many paths are hard-coded or derived from the current working directory rather than from a central config system.

Observed: The clearest primary execution path in the public repo is HEX patch-level inference via `hex/test_codex_lung_marker.py`. That script loads a MUSK-based regression model, builds a patch table from CSV metadata plus PNG patch files, runs batched mixed-precision inference, and writes per-patch predictions plus biomarker-level Pearson correlations. Strongly inferred: the broader slide-level HEX-to-MICA handoff is only partially exposed, because downstream utilities expect per-slide HDF5 prediction files that no visible script in this repo clearly writes.

## 2. Architecture

Observed: The repository has four practical layers.

- Preprocessing scripts at the repo root: `extract_marker_info_patch.py`, `extract_he_patch.py`, and `check_splits.py`.
- HEX model code in `hex/`: patch dataset construction, training, inference, and slide-level rasterization helpers.
- MICA model code in `mica/`: dataset loading, multimodal survival training/evaluation, and CODEX feature extraction utilities.
- Bundled artifacts/sample assets: `hex/checkpoint.pth`, `hex/model.safetensors`, `hex/sample_data/`, and `mica/tcga_splits/`.

Observed: The architectural center of gravity for HEX is split between `hex/test_codex_lung_marker.py`, `hex/train_dist_codex_lung_marker.py`, `hex/utils.py`, and `hex/hex_architecture.py`. The architectural center of gravity for MICA is `mica/train_mica.py`, `mica/core_utils.py`, `mica/dataset.py`, and `mica/models/model_coattn.py`.

Observed: There is no installable Python package structure (`__init__.py`, `pyproject.toml`, `setup.py`, etc.). Strongly inferred: the scripts are meant to be run from inside `hex/` or `mica/`, because imports such as `from utils import PatchDataset` and `from hex_architecture import CustomModel` are relative to the script directory rather than the repo root.

## 3. Key Components

- `HEX patch inference/evaluation` — `hex/test_codex_lung_marker.py` loads a checkpoint, enumerates patch metadata from `channel_registered/*.csv`, maps rows to PNG patch paths under `he_patches/`, runs the model, and writes `patch_predictions.csv` plus `biomarker_pearson_r.csv`. Observed.
- `HEX inference model definition` — `hex/hex_architecture.py` defines a MUSK-backed visual encoder plus a small regression head that outputs 40 biomarker values. `hex/musk_utils.py` handles weight loading/interpolation for this inference-specific model wrapper. Observed.
- `HEX training runtime` — `hex/train_dist_codex_lung_marker.py` and `hex/utils.py` implement distributed training with `torch.distributed`, mixed precision, adaptive robust loss, and per-marker FDS smoothing buffers. Observed.
- `HEX dataset/preprocessing bridge` — `extract_marker_info_patch.py` reads patch coordinates from HDF5 and computes per-patch mean marker intensities from CODEX OME-TIFF data; `extract_he_patch.py` cuts matching H&E PNG patches from WSIs; `check_splits.py` validates patient-level split integrity for both HEX and MICA. Observed.
- `HEX slide reconstruction helper` — `hex/virtual_codex_from_h5.py` reconstructs a dense slide-level virtual CODEX array from per-patch predictions stored as `codex_prediction` plus `coords` in HDF5. Observed.
- `MICA training/evaluation pipeline` — `mica/train_mica.py`, `mica/core_utils.py`, `mica/dataset.py`, and `mica/models/model_coattn.py` load H&E bag features (`pt_files/*.pt`) plus HEX-derived CODEX features (`features.h5`), train the MCAT-style co-attention survival model, and compute c-index metrics. Observed.
- `MICA CODEX feature extraction` — `mica/codex_h5_png2fea.py` converts per-slide HEX HDF5 predictions into dense `.npy` virtual CODEX images, then extracts per-channel DINOv2 embeddings and writes `features.h5` for MICA. Observed.
- `Bundled artifacts` — `hex/checkpoint.pth` and `hex/model.safetensors` ship with the repo, but only `hex/checkpoint.pth` is referenced directly by the public HEX test script. Observed.

## 4. Dependency Graph

- `extract_marker_info_patch.py` -> `palom`, `h5py`, CODEX OME-TIFF input, patch-coordinate HDF5 input
- `extract_he_patch.py` -> `openslide`, per-slide CSV labels, WSI `.svs` input
- `hex/train_dist_codex_lung_marker.py` -> `hex/utils.py`, `torch.distributed`, `robust_loss_pytorch`
- `hex/utils.py` -> external `musk.utils`, `timm.create_model`, `torchvision`, `scipy`
- `hex/test_codex_lung_marker.py` -> `hex/hex_architecture.py`, `hex/utils.py`, `torchvision`, `scipy.stats`
- `hex/hex_architecture.py` -> `hex/musk_utils.py`, `timm.create_model`
- `hex/virtual_codex_from_h5.py` -> `openslide`, `h5py`, per-slide HEX HDF5 predictions
- `mica/train_mica.py` -> `mica/dataset.py`, `mica/core_utils.py`, `mica/utils.py`
- `mica/core_utils.py` -> `mica/models/model_coattn.py`, `mica/utils.py`
- `mica/dataset.py` -> clinical CSV, split CSVs, H&E bag `.pt` files, CODEX `features.h5`
- `mica/codex_h5_png2fea.py` -> `openslide`, per-slide HEX HDF5 predictions, `facebookresearch/dinov2`
- `mica/test_mica.py` -> `mica/models/model_coattn.py`, Captum, but also absent imports such as `datasets.dataset_survival` and `file_utils`

## 5. Execution Flow

### Primary flow

1. `hex/test_codex_lung_marker.py` sets `save_dir`, derives `checkpoint_path`, and calls `load_model()`. Observed.
2. `load_model()` instantiates `hex/hex_architecture.CustomModel(visual_output_dim=1024, num_outputs=40)`, loads the checkpoint with `strict=False`, wraps it in `nn.DataParallel`, moves it to CUDA, and switches to eval mode. Observed.
3. The script reads patch-level marker CSVs from `hex/sample_data/channel_registered/`, optionally narrows evaluation to validation IDs from `hex/sample_data/splits_0.csv`, and constructs one row per patch with image paths like `he_patches/<slide>/<slide>_<index>.png`. Observed.
4. `PatchDataset` from `hex/utils.py` loads each PNG patch and its 40 `mean_intensity_channel*` labels. The evaluation transform resizes to `384x384`, converts to tensor, and normalizes with ImageNet Inception statistics. Observed.
5. The model forward path in `hex/hex_architecture.py` sends the image tensor through a MUSK visual encoder (`with_head=False`, `out_norm=False`), then through a two-layer MLP regression head and final linear layer to produce a 40-value prediction vector. Observed.
6. The script accumulates predictions and labels across the dataloader, writes one CSV row per patch to `patch_predictions.csv`, computes per-biomarker Pearson correlations, and writes `biomarker_pearson_r.csv`. Observed.

### Secondary flows

- `HEX training` — `hex/train_dist_codex_lung_marker.py` initializes NCCL DDP, constructs train/val patch tables from the same CSV-plus-PNG layout, trains `utils.CustomModel` with adaptive robust loss and optional FDS smoothing, logs TensorBoard summaries, and saves checkpoints every 5 epochs. Observed.
- `HEX slide reconstruction` — `hex/virtual_codex_from_h5.py` expects one HDF5 file per slide containing `codex_prediction` and `coords`, maps predictions back into a dense `[H, W, C]` NumPy array using WSI magnification heuristics, and saves `<slide>.npy`. Observed.
- `MICA feature preparation` — `mica/codex_h5_png2fea.py` performs a similar slide reconstruction, converts each of the 40 channels into a grayscale RGB image, extracts DINOv2 features per channel, and writes one dataset per slide into `features.h5`. Observed.
- `MICA training` — `mica/train_mica.py` builds paths from `--base_path` and `--project_name`, loads split CSVs and clinical labels through `mica/dataset.py`, trains `MCAT_Surv`, and stores fold checkpoints/results under `results_dir`. Observed.
- `MICA evaluation` — `mica/test_mica.py` intends to reload trained MICA checkpoints and compute c-index values, optionally with integrated gradients. Strongly inferred: this path is stale or incomplete in the public repo because its imports do not match the files actually present. Evidence: it imports `datasets.dataset_survival`, `file_utils`, and calls `model.captum`, none of which are defined in the visible repository.

### Notebooks

Observed: No notebooks were present in the repository scan.

## 6. Data Flow

Observed: The HEX data path starts before model runtime. `extract_marker_info_patch.py` reads patch coordinates from HDF5 and uses a registered CODEX OME-TIFF to compute mean channel intensities per patch, producing CSVs with `x`, `y`, and `mean_intensity_channel1..40`. `extract_he_patch.py` then uses those coordinates to cut matching H&E PNG patches from WSIs.

Observed: During HEX training/inference, each row in the CSV is converted into:

- an image path under `he_patches/<slide>/...png`
- a 40-dimensional floating-point label vector from `mean_intensity_channel1..40`
- an implicit patch identifier built from `slide` and `index`

Observed: The HEX model input is a normalized RGB patch tensor resized to `384x384`. The immediate model output is:

- a 40-dimensional biomarker prediction vector
- a 128-dimensional intermediate feature vector from the regression head

Observed: Public HEX evaluation persists patch-level outputs as CSV tables. Strongly inferred: downstream slide-level utilities expect a separate step that groups patch predictions by slide and saves them with original coordinates in HDF5. Evidence: both `hex/virtual_codex_from_h5.py` and `mica/codex_h5_png2fea.py` read `codex_prediction` plus `coords`, but no visible script writes that pair.

Observed: The MICA data path combines two modalities per slide:

- H&E bag features from `pt_files/<slide>.pt`, loaded with `torch.load()` in `mica/dataset.py`
- HEX-derived per-slide CODEX features loaded from `features.h5`, where each slide stores a `(40, feature_dim)` array

Observed: `mica/models/model_coattn.py` projects H&E bag features from 1024 to 256 dimensions and CODEX features from 384 to 256 dimensions, applies co-attention plus per-modality transformers/pooling, fuses the modality embeddings, and outputs discrete survival hazards and cumulative survival values.

## 7. Configuration

Observed: There is no central config file system. Configuration is split across hard-coded script variables, optional CSV files, and `argparse` in the MICA scripts.

Observed: HEX uses mostly in-script configuration:

- `hex/test_codex_lung_marker.py` hard-codes `save_dir`, `data_dir`, batch size, worker count, image size, and the expectation that `checkpoint.pth` lives under `save_dir`.
- `hex/train_dist_codex_lung_marker.py` hard-codes `save_dir`, `data_dir`, optimizer/scheduler settings, epoch count, image size, and FDS settings.
- `hex/sample_data/splits_0.csv` is used opportunistically when present to define validation IDs.

Observed: MICA uses `argparse`, but some parsed settings are overwritten in code:

- `args.which_splits = '5foldcv'`
- `args.k_start = 0`, `args.k_end = 5`
- path derivations from `args.base_path` and `args.project_name`

Strongly inferred: precedence is effectively "script constants and post-parse mutations first, CLI second, file-based splits/data third." Evidence: several MICA arguments are parsed and then replaced before the main run begins.

Observed: External data layout is part of the configuration contract. For example, MICA expects:

- `<base>/<PROJECT>/splits/splits_<fold>.csv`
- `<base>/<PROJECT>/features/pt_files/<slide>.pt`
- `<base>/<PROJECT>/he2codex/fea_files/features.h5`
- `<base>/<PROJECT>/<project_name>_clin.csv`

## 8. Environment Setup

### Documented

Observed: The user provided the authoritative activation command for this workspace: `source ~/activate_conda.sh; conda activate hex`.

Observed: `README.md` documents a GPU-oriented environment centered on Python 3.10, PyTorch 2.4.0+cu118, CUDA 11.8/cuDNN 9.1, and a long list of libraries including `accelerate`, `captum`, `h5py`, `openslide-python`, `timm`, `torch-geometric`, `transformers`, plus external upstream projects such as MUSK, Palom, CLAM, MCAT, and DINOv2.

### Inferred

Observed: The repo does not include `environment.yml`, `requirements.txt`, `pyproject.toml`, or `setup.py`, so the README plus user instruction are the only explicit setup sources in-repo.

Observed: A minimal read-only import check inside the provided `hex` conda env found `torch`, `timm`, and `h5py`, but did not resolve `musk`, `openslide`, or `transformers`.

Strongly inferred: the supplied environment is not yet sufficient for every public script as written, or some dependencies require extra path/package setup. Evidence: HEX training imports `musk`, HEX/MICA slide utilities import `openslide`, and `mica/codex_h5_png2fea.py` imports `transformers`, but those modules were not found in the minimal env check.

Strongly inferred: scripts are intended to be run from within `hex/` or `mica/`, not from the repo root, unless `PYTHONPATH` is adjusted. Evidence: imports such as `from utils import ...` and `from hex_architecture import ...` are local-directory imports rather than package imports.

## 9. Optional Pipelines

- `Preprocessing pipeline` — registers CODEX with H&E externally, extracts patch-level marker intensities (`extract_marker_info_patch.py`), cuts matching H&E patches (`extract_he_patch.py`), and validates split CSVs (`check_splits.py`). Observed.
- `HEX training pipeline` — `hex/train_dist_codex_lung_marker.py` trains a patch-level regression model on PNG patches plus 40-channel numeric labels. Observed.
- `HEX inference/evaluation pipeline` — `hex/test_codex_lung_marker.py` evaluates a checkpoint on patch-level data and emits CSV metrics/results. Observed.
- `Virtual CODEX reconstruction pipeline` — `hex/virtual_codex_from_h5.py` and the first half of `mica/codex_h5_png2fea.py` convert sparse patch predictions back into dense slide-level arrays. Observed.
- `MICA multimodal survival pipeline` — `mica/train_mica.py` trains `MCAT_Surv` on H&E bag features plus DINOv2 features extracted from HEX-derived CODEX channels. Observed.

## 10. Other Notes

Observed: `hex/` contains a local `checkpoint.pth` and `model.safetensors`. The public HEX test script references `checkpoint.pth`, but the local MUSK weights file is not wired explicitly into the visible inference script.

Observed: `README.md` describes the repo as containing both HEX and MICA, and that matches the visible folder structure.

Observed: Some public commands in `README.md` omit directory context. For example, the visible HEX scripts live under `hex/` and use local-directory imports, so running them from the repo root exactly as written would likely fail without changing directories or adjusting `PYTHONPATH`.

## 11. Uncertainties

- Unknown: the exact supported way to load MUSK backbone weights for HEX inference in the public repo. `hex/hex_architecture.py` uses `hex/musk_utils.py` with `"hf_hub:xiangjx/musk"`, but the local helper appears to call `safetensors.load_file()` on that path, while a local `hex/model.safetensors` also exists and is not referenced.
- Unknown: which script is meant to create the per-slide HDF5 files containing `codex_prediction` and `coords` for downstream slide reconstruction. The visible HEX evaluation script writes CSV outputs, not HDF5 outputs.
- Unknown: whether `mica/test_mica.py` is expected to run unmodified in this public repo. Its imports (`datasets.dataset_survival`, `file_utils`) and `model.captum()` call do not match the visible file set.
- Unknown: whether `hex/checkpoint.pth` is only the demo pipeline checkpoint or a general pretrained release checkpoint. The README refers to demo checkpoints, but the exact status of the bundled file is not encoded in code.

## 12. Coverage Report

- docs read: `README.md`, provided `repo-study-overview` skill instructions
- config files read: no dedicated config files present; inspected hard-coded config/argparse logic in `hex/test_codex_lung_marker.py`, `hex/train_dist_codex_lung_marker.py`, `mica/train_mica.py`, `mica/test_mica.py`
- core code inspected: `hex/test_codex_lung_marker.py`, `hex/hex_architecture.py`, `hex/train_dist_codex_lung_marker.py`, `hex/utils.py`, `hex/musk_utils.py`, `hex/virtual_codex_from_h5.py`, `mica/train_mica.py`, `mica/test_mica.py`, `mica/dataset.py`, `mica/core_utils.py`, `mica/models/model_coattn.py`, `mica/codex_h5_png2fea.py`
- preprocessing/support code inspected: `extract_marker_info_patch.py`, `extract_he_patch.py`, `check_splits.py`
- notebooks inspected: none present
- tests inspected: no dedicated test suite present; inspected script-style evaluation files `hex/test_codex_lung_marker.py` and `mica/test_mica.py`
- skipped or lightly sampled: the long custom attention helper internals in `mica/models/model_coattn.py`, bundled split CSVs under `mica/tcga_splits/`, and sample data files beyond confirming their existence/shape role

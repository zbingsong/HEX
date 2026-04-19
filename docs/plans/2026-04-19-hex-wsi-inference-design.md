# HEX WSI Inference Design

## Summary

Build a new standalone HEX inference script that takes one `.svs` whole-slide image and produces virtual spatial proteomics from sliding-window inference. The first version targets one WSI level at a time, but the design keeps level handling generic so the same code can later emit predictions for multiple WSI levels.

The script will run HEX on centered `224 x 224` patches sampled on a dense stride-4 grid. Each output location represents one model prediction for a patch centered at that level-space coordinate. Predictions will be written to a per-level `.npy` file and also exported into an OME-TIFF. Background patches that are all-white or near-white should be skipped to avoid wasting compute.

## Goals

- Accept a single `.svs` input file and a HEX checkpoint path.
- Run inference on a requested WSI level, defaulting to the top level.
- Sample centered `224 x 224` RGB patches with stride `4`.
- Pad patches symmetrically at all borders, including the top-left corner.
- Skip near-white patches using a configurable threshold rule.
- Save one `.npy` prediction tensor per inferred level.
- Save one skip-mask `.npy` per inferred level for QC.
- Export one OME-TIFF containing all inferred levels. For now this will usually contain one level.

## Non-Goals

- No MICA integration.
- No tissue segmentation model in v1.
- No direct multi-level inference implementation in v1, only a structure that is easy to extend.
- No attempt to match the repo’s missing HDF5 slide-prediction intermediate format.

## Inputs

- Source WSI: `.svs`
- HEX checkpoint: expected to load through the existing HEX model definition
- Parameters:
  - `level`
  - `patch_size` default `224`
  - `stride` default `4`
  - `batch_size`
  - `num_workers`
  - background filter thresholds
  - output directory

## Core Sampling Semantics

- Output pixel `(r, c)` corresponds to a patch centered at level-space coordinate `(c * stride, r * stride)`.
- With `patch_size = 224`, the requested crop spans:
  - `x = center_x - 112 : center_x + 112`
  - `y = center_y - 112 : center_y + 112`
- If any part of that crop lies outside the WSI level, pad to keep the model input exactly `224 x 224`.
- This rule applies at all borders, including the first sample at the top-left corner.

## Output Semantics

- Per-level prediction tensor shape:
  - approximately `ceil(H / stride) x ceil(W / stride) x 40`
- Channel order follows the existing HEX biomarker order used by `hex/test_codex_lung_marker.py`
- Per-level skip mask shape:
  - `ceil(H / stride) x ceil(W / stride)`
- File outputs:
  - `<slide_id>_level<k>_pred.npy`
  - `<slide_id>_level<k>_skipmask.npy`
  - `<slide_id>.ome.tiff`

## Architecture

### 1. CLI script

Add a new script under `hex/` that:

- parses arguments
- opens the WSI
- validates the requested level
- loads the HEX model
- runs batched sliding-window inference
- writes `.npy` outputs
- writes OME-TIFF output

### 2. Sliding-window dataset

Define an on-the-fly dataset for one WSI level:

- precomputes output-grid dimensions
- maps dataset index to output-grid row/column
- converts row/column to patch-center coordinates
- reads the patch from OpenSlide
- pads out-of-bounds regions
- computes the near-white skip decision
- returns either:
  - a normalized tensor plus output-grid coordinates, or
  - a skip marker plus output-grid coordinates

### 3. Model loading

Reuse the existing HEX inference model path:

- `hex/hex_architecture.py`
- same normalization as `hex/test_codex_lung_marker.py`
- same 40-channel output interpretation

### 4. Dense raster writer

Use disk-backed writing rather than holding the full output in RAM:

- write predictions into a memory-mapped array on disk
- write skip mask into a second disk-backed array
- flush regularly

### 5. OME-TIFF export

After `.npy` generation:

- load or stream the saved per-level arrays
- convert them into an OME-TIFF layout
- attach channel names and level metadata

The exporter should accept a list of per-level arrays even if the first version only uses one level.

## Background Skipping

Use a cheap configurable heuristic:

- classify a patch as background if at least `p%` of pixels exceed threshold `t` in all channels
- recommended defaults:
  - `t = 240`
  - `p = 0.98`

For skipped patches:

- do not run HEX
- write a default background prediction vector
- write `True` in the skip mask

Default background prediction should be configurable. The initial recommendation is zeros.

## Data Flow

1. Open WSI and read `level_dimensions[level]`.
2. Compute output-grid size from level dimensions and stride.
3. Iterate over output-grid locations.
4. Convert each location to a centered patch request.
5. Read and pad the RGB patch.
6. Apply the background filter.
7. Batch only non-skipped patches.
8. Normalize images and run HEX.
9. Write prediction vectors into the dense raster at the corresponding output-grid coordinates.
10. Write skip decisions into the mask.
11. Save `.npy` artifacts.
12. Export OME-TIFF.

## Error Handling

- Fail fast if the WSI path does not exist.
- Fail fast if the level index is invalid.
- Fail fast if the checkpoint fails to load.
- Treat OpenSlide read failures as hard errors by default.
- If OME-TIFF export fails after `.npy` files are written, keep the `.npy` files and report the export failure separately.

## Performance Constraints

- Top-level WSI inference with stride `4` can still be very large.
- Use batched inference plus disk-backed outputs.
- Keep the design open to later additions such as:
  - ROI restriction
  - tissue masks
  - multi-level loops
  - resumable chunk processing

## Verification Strategy

- Smoke test on a small crop or small WSI level.
- Verify output tensor shape matches `ceil(H / stride) x ceil(W / stride) x 40`.
- Verify `(0, 0)` uses centered top-left padded sampling.
- Verify right and bottom borders are padded correctly.
- Verify near-white patches are skipped and marked.
- Verify the OME-TIFF channel count and metadata match the `.npy` outputs.

## Open Questions Resolved

- Output style: dense stride-space raster, not sparse patch tables.
- Border rule: centered patch sampling with symmetric padding at all borders, including the top-left corner.
- Background handling: skip near-white patches.
- Initial scope: top-level only, but code should be easy to extend to more levels.

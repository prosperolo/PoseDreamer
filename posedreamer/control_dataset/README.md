# Control training dataset

Builds the ~130K (control render, real image) pairs used to train the spatial
control LoRA (paper §3.1), from two annotated sources:

## DensePose-COCO

1. `get_data.sh` — downloads the DensePose-COCO CSE annotations into the
   untracked `DensePose_COCO/` folder (see also `../../IMPORTED_REPOS.md`).
2. `save_densepose_coco_crops.py` (launcher: `densepose_coco.sh`) — renders
   the CSE annotations into dense control images and matching image crops:
   `--output_dir` receives `densepose_changes/` and `image_crops/`.

## AGORA

- `agora_render.py` (launcher: `agora_crops.sh`) — renders AGORA's
  ground-truth SMPL-X meshes into control images + crops, one job per
  dataframe split. Requires the AGORA images, dataframes and neutral SMPL-X
  ground truth from https://agora.is.tue.mpg.de/.

The resulting pairs are assembled into the `dataset.json` consumed by
[`../generation/train_spatial.sh`](../generation/train_spatial.sh).

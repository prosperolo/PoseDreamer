# Imported repositories

This project depends on a few third-party repositories that are cloned into
this tree but are **not tracked by git**. Each is excluded via `.gitignore`.
When setting up the project from scratch, clone the listed repos into the
paths below.

## `smplx/`

Source: https://github.com/vchoutas/smplx

Used by the SMPL-X model and the `transfer_model` package (SMPL-X ↔ SMPL
conversion). Our wrapper scripts that depend on it live in
[`posedreamer/smplx_convert/`](posedreamer/smplx_convert/) — see that folder's
README for invocation.

Clone:
```bash
git clone https://github.com/vchoutas/smplx.git smplx
```

After cloning, place the SMPL/SMPL-X model files and the deformation-transfer
pickles in `posedreamer/weights/` (see [`posedreamer/weights/README.txt`](posedreamer/weights/README.txt)).

## `posedreamer/control_dataset/agora_evaluation/`

Source: https://github.com/pixelite-lab/agora_evaluation

(If not present locally, that's fine — it's only needed for AGORA-format
evaluation.)

## `EasyControl/`

Source: https://github.com/Xiaojiu-z/EasyControl

Inference / training framework used by the FLUX-based image generation
pipeline. Our wrapper scripts live in
[`posedreamer/generation/`](posedreamer/generation/) and call into the
cloned repo for the model code.

Clone:
```bash
git clone https://github.com/Xiaojiu-z/EasyControl.git EasyControl
```

See [`posedreamer/generation/README.md`](posedreamer/generation/README.md)
for invocation details.

## `posedreamer/control_dataset/DensePose_COCO/`

Not a repository but downloaded data, handled the same way: the folder is
untracked (see `.gitignore`) and holds the DensePose-COCO CSE annotations
(`densepose_train2014_cse.json`). Populate it by running

```bash
cd posedreamer/control_dataset && ./get_data.sh
```

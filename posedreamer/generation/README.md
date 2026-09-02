# Generation (FLUX + EasyControl) and DPO alignment

Drives controllable image generation (paper §3.1) and Direct Preference
Optimization of the control model (paper §3.2).

The FLUX/EasyControl model code lives in the **`EasyControl/` clone at the
repo root** — see [`../../IMPORTED_REPOS.md`](../../IMPORTED_REPOS.md). Clone
it before running anything here; the launchers add it to `PYTHONPATH`.

## Spatial control LoRA

- `train_spatial.sh` — trains the mesh-to-RGB spatial control LoRA with
  EasyControl's trainer on the pairs from
  [`../control_dataset/`](../control_dataset/README.md).
- `infer_folder.py` — batch inference over a folder of control renders.

## DPO alignment (`alignment/`)

1. `infer_for_dpo.py` (launcher: `generate_fb.sh`) — generates candidate
   images per control render for preference-pair construction; samples are
   scored with OKS (`../label_generation/compute_oks_metric.py`), and
   `alignment/merge_metadata.py` merges the scores into training metadata.
2. `alignment/train.py` (launcher: `alignment/scripts/train.sh`, config:
   `alignment/config/train.yaml`) — Flow-DPO training of a rank-128 LoRA on
   OKS-ranked preference pairs (`alignment/dataset.py`, `alignment/loss.py`,
   `alignment/trainer.py`).
3. `alignment/inference.py` / `alignment/evaluate_single_checkpoint.py`
   (launchers: `alignment/scripts/{inference,evaluate}.sh`) — run and score
   DPO checkpoints.
4. `infer_w_dpo.py` — final large-scale dataset generation with the control
   LoRA + DPO LoRA stacked.
5. `alignment/generate_samples.py` / `alignment/generate_variations.py`
   (launchers in `alignment/scripts/`) — qualitative grids and caption
   variations for a fixed control image.

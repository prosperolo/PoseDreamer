#!/bin/bash
# Render AGORA ground-truth SMPL-X meshes into control images + crops.
# Example SLURM array launcher (one task per dataframe split); adjust
# resources/paths for your cluster, then: sbatch agora_crops.sh
#SBATCH --job-name=agora_crops
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3%4

AGORA_ROOT=/path/to/AGORA
OUTPUT_DIR=/path/to/control-dataset-agora

cd "$(dirname "$0")" || exit
SPLIT_ID=$(printf "%d" ${SLURM_ARRAY_TASK_ID:-0})
echo "Running job with split ID: split_$SPLIT_ID"

python agora_render.py \
  --dataframe "$AGORA_ROOT/train_df/SMPLX/pt$SPLIT_ID" \
  --img_dir "$AGORA_ROOT/train_images" \
  --model_path ../weights \
  --smplx_neutral_gt "$AGORA_ROOT/smplx_gt_neutral" \
  --kid_template_path ../weights/smplx_kid_template.npy \
  --output_dir "$OUTPUT_DIR"

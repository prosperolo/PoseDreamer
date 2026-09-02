#!/bin/bash
# Generate images for DPO preference pairs with the spatial-control LoRA.
# Example SLURM array launcher (tasks shard the control-render list).
# Requires the EasyControl clone at the repo root (see IMPORTED_REPOS.md).
#SBATCH --job-name=generate_body
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%4

CONTROL_LORA=/path/to/models/supervised/checkpoint-XXXX/lora.safetensors
DENSEPOSE_DIR=/path/to/densepose-renders/
METADATA_DIR=/path/to/annotations/metadata/
OUT_IMAGES=/path/to/generated-images-dpo/
OUT_METADATA=/path/to/generated-metadata-dpo/

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/EasyControl:$REPO_ROOT:$PYTHONPATH"

python "$REPO_ROOT/posedreamer/generation/infer_for_dpo.py" \
  "$CONTROL_LORA" "$DENSEPOSE_DIR" "$METADATA_DIR" "$OUT_IMAGES" "$OUT_METADATA"

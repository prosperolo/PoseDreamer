#!/bin/bash
# Generate caption variations for one control image (Fig. 1-style grid).
# Example SLURM launcher; adjust resources/paths for your cluster.
#SBATCH --job-name=variations_dpo
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")/.." || exit
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/EasyControl:$REPO_ROOT:$PYTHONPATH"

BASE_MODEL="black-forest-labs/FLUX.1-dev"  # HF id or local FLUX.1-dev path
CONTROL_LORA=/path/to/models/supervised/checkpoint-XXXX/lora.safetensors
CONTROL_IMAGE=/path/to/densepose-renders/example.png
DPO_CHECKPOINT=/path/to/oks-dpo-training/run/checkpoint.ckpt
SAVE_FOLDER=/path/to/image-variations

python generate_variations.py "$CONTROL_IMAGE" "$BASE_MODEL" "$CONTROL_LORA" \
  "$DPO_CHECKPOINT" "$SAVE_FOLDER"

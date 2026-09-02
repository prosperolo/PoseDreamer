#!/bin/bash
# Batch inference with the DPO-aligned control model.
# Example SLURM launcher; adjust resources/paths for your cluster.
#SBATCH --job-name=inference_dpo
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")/.." || exit
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/EasyControl:$REPO_ROOT:$PYTHONPATH"

BASE_MODEL="black-forest-labs/FLUX.1-dev"  # HF id or local FLUX.1-dev path
CONTROL_LORA=/path/to/models/supervised/checkpoint-XXXX/lora.safetensors
DPO_RUN_DIR=/path/to/oks-dpo-training/run     # folder of DPO checkpoints
DENSEPOSE_DIR=/path/to/densepose-renders/
METADATA_DIR=/path/to/annotations/metadata
SAVE_FOLDER=/path/to/dpo-generations
NUM_SAMPLES=10

python inference.py "$BASE_MODEL" "$CONTROL_LORA" "$DPO_RUN_DIR" \
  "$DENSEPOSE_DIR" "$METADATA_DIR" "$SAVE_FOLDER" "$NUM_SAMPLES"

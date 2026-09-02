#!/bin/bash
# Generate qualitative samples from the (optionally DPO-tuned) control model.
# Example SLURM launcher; adjust resources/paths for your cluster.
#SBATCH --job-name=generate_samples
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")/.." || exit
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/EasyControl:$REPO_ROOT:$PYTHONPATH"

BASE_MODEL="black-forest-labs/FLUX.1-dev"  # HF id or local FLUX.1-dev path
CONTROL_LORA=/path/to/models/supervised/checkpoint-XXXX/lora.safetensors
DENSEPOSE_DIR=/path/to/densepose-renders/
METADATA_DIR=/path/to/annotations/metadata
SAVE_FOLDER=/path/to/sample-generations

# Optional: DPO checkpoint to apply (leave empty for the base control model)
CHECKPOINT_PATH=""
# Optional: add pose control
USE_POSE_CONTROL="true"

python generate_samples.py \
    --base_model_path="$BASE_MODEL" \
    --control_lora_path="$CONTROL_LORA" \
    --densepose_path="$DENSEPOSE_DIR" \
    --metadata_path="$METADATA_DIR" \
    --save_folder="$SAVE_FOLDER" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --use_pose_control="$USE_POSE_CONTROL" \
    --num_samples=10 \
    --guidance_scale=3.5 \
    --num_inference_steps=25 \
    --max_sequence_length=512

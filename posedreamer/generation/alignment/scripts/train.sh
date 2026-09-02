#!/bin/bash
# Train the DPO alignment LoRA (config: config/train.yaml).
# Example SLURM launcher; adjust resources/paths for your cluster.
#SBATCH --job-name=train_dpo
#SBATCH --time=3-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")/.." || exit
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/EasyControl:$REPO_ROOT:$PYTHONPATH"

BASE_MODEL="black-forest-labs/FLUX.1-dev"  # HF id or local FLUX.1-dev path
CONTROL_LORA=/path/to/models/supervised/checkpoint-XXXX/lora.safetensors

# Model/data paths are set in config/train.yaml; override on the CLI if needed:
python train.py model.pretrained_model_name_or_path="$BASE_MODEL" model.control_lora_path="$CONTROL_LORA"

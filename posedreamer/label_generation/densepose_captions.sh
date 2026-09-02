#!/bin/bash
# Caption AMASS control renders with a VLM (pose + scene descriptions).
# Example SLURM array launcher (tasks shard the image list).
#SBATCH --job-name=densepose_captions
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

IMAGES_FOLDER=/path/to/densepose-renders
OUT_DIR=/path/to/captions

cd "$(dirname "$0")" || exit
python densepose_captions.py --images_folder="$IMAGES_FOLDER" --out_dir="$OUT_DIR"

#!/bin/bash
# Process LAION images: YOLO person detection, crops, DensePose, captions,
# metadata. Example SLURM array launcher (one task per LAION split).
#SBATCH --job-name=process_laion
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%8

LAION_PATH=/path/to/laion_face_data
SAVE_DIR=/path/to/laion-annotations
SPLIT_OFFSET=0  # first LAION split index handled by array task 0

cd "$(dirname "$0")" || exit
SPLIT_ID=$(printf "%05d" $((${SLURM_ARRAY_TASK_ID:-0} + SPLIT_OFFSET)))
echo "Running job with split ID: split_$SPLIT_ID"

python process_laion.py \
  --laion_path="$LAION_PATH" \
  --save_dir="$SAVE_DIR" \
  --split=split_$SPLIT_ID

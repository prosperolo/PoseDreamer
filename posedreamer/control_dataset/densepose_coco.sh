#!/bin/bash
# Render DensePose-COCO CSE annotations into control images + crops
# (run ./get_data.sh first to fetch the annotations).
# Example SLURM launcher; adjust resources for your cluster.
#SBATCH --job-name=densepose_coco_crops
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

OUTPUT_DIR=/path/to/control-dataset-coco

cd "$(dirname "$0")" || exit
python save_densepose_coco_crops.py --output_dir "$OUTPUT_DIR"

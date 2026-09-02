#!/bin/bash
# Caption the LAION crops produced by process_laion.sh with a VLM.
# Example SLURM array launcher.
#SBATCH --job-name=generate_captions
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%8

DATASET_FOLDER=/path/to/laion-annotations

cd "$(dirname "$0")" || exit
python generate_captions.py --dataset_folder="$DATASET_FOLDER"

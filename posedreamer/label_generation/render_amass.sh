#!/bin/bash
# Render AMASS motion-capture frames into SMPL-X control images + labels.
# Example SLURM array launcher (tasks shard the sequence list).
#SBATCH --job-name=render_amass
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15%16

AMASS_ROOT=/path/to/AMASS
OUT_ROOT=/path/to/amass-renders

cd "$(dirname "$0")" || exit
python render_amass.py --input_data_root="$AMASS_ROOT" --out_data_root="$OUT_ROOT"

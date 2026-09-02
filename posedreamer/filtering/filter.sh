#!/bin/bash
# Example SLURM array launcher for the filtering pipeline.
# Adjust partition / resources / paths for your cluster, then:
#   sbatch --array=0-11 filter.sh
#SBATCH --job-name=posedreamer_filter
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")" || exit

python filter.py \
  action=copy \
  action.dest_base_dir=/path/to/filtered-output

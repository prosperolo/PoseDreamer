#!/bin/bash
# Launch parallel AMASS npz conversion with 4 processes per GPU

# 1. Check if GPUs were provided
if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_list>"
  echo "Example: $0 0,1,2,3"
  exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_PROCESSES_PER_GPU=1
ROOT_DIR="${2:?Usage: $0 <gpu_list> <root_dir> [smplx_model_folder] [exp_cfg] [batch_size] [files_per_batch]}"
SMPLX_MODEL_FOLDER="${3:-models}"
EXP_CFG="${4:-config_files/smplx2smpl.yaml}"
BATCH_SIZE="${5:-192}"           # Number of frames to process in parallel
FILES_PER_BATCH="${6:-16}"       # Number of files to batch together (16 files × 12 frames = 192)
LOG_DIR="${SCRIPT_DIR}/log"

# Create log directory
mkdir -p "$LOG_DIR"

# 2. Split comma-separated string (e.g., "0,1,2,3") into array
IFS=',' read -ra GPUS <<< "$1"

# 3. Launch processes
HOST=$(hostname)
PROCESS_ID=0

for GPU_ID in "${GPUS[@]}"; do
    echo "Launching $NUM_PROCESSES_PER_GPU processes on GPU $GPU_ID..."
    
    for PROC_NUM in $(seq 1 $NUM_PROCESSES_PER_GPU); do
        LOG_FILE="${LOG_DIR}/convert_${HOST}_gpu${GPU_ID}_proc${PROC_NUM}.log"
        
        echo "  - Process $PROC_NUM on GPU $GPU_ID -> $LOG_FILE"
        
        # Launch in background with unique GPU assignment
        nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python convert.py \
            --root-dir "$ROOT_DIR" \
            --smplx-model-folder "$SMPLX_MODEL_FOLDER" \
            --exp-cfg "$EXP_CFG" \
            --batch-size "$BATCH_SIZE" \
            --files-per-batch "$FILES_PER_BATCH" \
            > "$LOG_FILE" 2>&1 &
        
        PROCESS_ID=$((PROCESS_ID + 1))
    done
done

echo "Launched $PROCESS_ID processes across ${#GPUS[@]} GPUs"
echo "Logs are in: $LOG_DIR"
echo "Monitor progress with: tail -f $LOG_DIR/convert_*.log"

# Example usage:
# bash convert.sh 0,1,2,3 /path/to/data
# bash convert.sh 0,1,2,3 /path/to/data models config.yaml 96 8
#
# Parameters:
#   $1: GPU list (required, e.g., "0,1,2,3")
#   $2: Root directory containing the .h5 files to convert (required)
#   $3: SMPLX model folder (optional, default: models)
#   $4: Config file (optional, default: config_files/smplx2smpl.yaml)
#   $5: Batch size (optional, default: 96 frames)
#   $6: Files per batch (optional, default: 8 files)

#!/bin/bash
# Train the spatial-control LoRA with EasyControl (clone it at the repo root
# first — see IMPORTED_REPOS.md). TRAIN_DATA is the dataset.json produced from
# the control dataset (source/target/caption columns).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export MODEL_DIR="black-forest-labs/FLUX.1-dev"  # HF id or local FLUX.1-dev path
export OUTPUT_DIR="./models/supervised"
export CONFIG="./default_config.yaml"
export TRAIN_DATA=/path/to/control-dataset/dataset.json
export LOG_PATH="$OUTPUT_DIR/log"
SPATIAL_TEST_IMAGE=/path/to/validation/test_color.png

cd "$REPO_ROOT/EasyControl/train" || exit

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "A DSLR picture of a football player." \
    --num_train_epochs=1000 \
    --validation_steps=1000 \
    --checkpointing_steps=1000 \
    --spatial_test_images "$SPATIAL_TEST_IMAGE" \
    --subject_test_images None \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=2

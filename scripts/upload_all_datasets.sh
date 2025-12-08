#!/bin/bash

# Upload all dataset combinations to wfang11's HuggingFace
# Models: Llama-3.1-70B-Instruct, Llama-3.1-8B-Instruct
# Datasets: MATH500, GPQA, MMLU
# Split: 1-5-94 (dev/val/test)

set -e  # Exit on error

echo "=========================================================================="
echo "Batch Upload: All Dataset Combinations to wfang11's HuggingFace"
echo "=========================================================================="
echo ""

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate weaver environment
echo "Activating weaver conda environment..."
conda activate weaver

# Configuration
TRAIN_SIZE=0.01
VAL_SIZE=0.05
TEST_SIZE=0.94
SEED=422  # Match amyguan's seed for identical splits

# Dataset configurations
# Format: "source_dataset|hub_name|display_name"
DATASETS=(
    "hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1|wfang11/math500-llama70b-1-5-94|MATH500 + Llama-70B"
    "hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1|wfang11/math500-llama8b-1-5-94|MATH500 + Llama-8B"
    "hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1|wfang11/gpqa-llama70b-1-5-94|GPQA + Llama-70B"
    "hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1|wfang11/gpqa-llama8b-1-5-94|GPQA + Llama-8B"
    "hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1|wfang11/mmlu-llama70b-1-5-94|MMLU + Llama-70B"
    "hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1|wfang11/mmlu-llama8b-1-5-94|MMLU + Llama-8B"
)

TOTAL=${#DATASETS[@]}
CURRENT=0

echo "Starting upload of $TOTAL dataset combinations (18 total datasets)..."
echo "Split: ${TRAIN_SIZE}-${VAL_SIZE}-${TEST_SIZE} (dev/val/test)"
echo ""

for dataset_config in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse configuration
    IFS='|' read -r SOURCE_DATASET HUB_NAME DISPLAY_NAME <<< "$dataset_config"
    
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$CURRENT/$TOTAL] Processing: $DISPLAY_NAME"
    echo "----------------------------------------------------------------------"
    echo "  Source: $SOURCE_DATASET"
    echo "  Target: $HUB_NAME"
    echo ""
    
    # Run the splitting script
    python scripts/split_and_upload_dataset.py \
        --dataset_name "$SOURCE_DATASET" \
        --hub_name "$HUB_NAME" \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --test_size $TEST_SIZE \
        --seed $SEED
    
    echo ""
    echo "✓ Completed: $DISPLAY_NAME"
    echo "  - ${HUB_NAME}-dev"
    echo "  - ${HUB_NAME}-val"
    echo "  - ${HUB_NAME}-test"
    
    # Small delay between uploads
    if [ $CURRENT -lt $TOTAL ]; then
        echo ""
        echo "Waiting 5 seconds before next upload..."
        sleep 5
    fi
done

echo ""
echo "=========================================================================="
echo "✓ ALL UPLOADS COMPLETE!"
echo "=========================================================================="
echo ""
echo "Successfully uploaded $TOTAL dataset combinations (18 total datasets)"
echo ""
echo "View your datasets at: https://huggingface.co/wfang11"
echo ""

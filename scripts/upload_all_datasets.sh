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
# Format: "source_dataset|hub_name|display_name|id_strategy|ref_dataset|id_prefix"
DATASETS=(
    # MATH500 70B: Already has IDs
    "hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1|wfang11/math500-llama70b-1-5-94|MATH500 + Llama-70B|none||"
    
    # MATH500 8B: Map from 70B
    "hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1|wfang11/math500-llama8b-1-5-94|MATH500 + Llama-8B|map|hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1|"
    
    # GPQA: Generate IDs (index) - Order matches between 70B/8B
    "hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1|wfang11/gpqa-llama70b-1-5-94|GPQA + Llama-70B|index||gpqa"
    "hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1|wfang11/gpqa-llama8b-1-5-94|GPQA + Llama-8B|index||gpqa"
    
    # MMLU: Generate IDs (index) - Order DOES NOT match, IDs will be unique but not aligned across models
    "hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1|wfang11/mmlu-llama70b-1-5-94|MMLU + Llama-70B|index||mmlu"
    "hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1|wfang11/mmlu-llama8b-1-5-94|MMLU + Llama-8B|index||mmlu"
)

TOTAL=${#DATASETS[@]}
CURRENT=0

echo "Starting upload of $TOTAL dataset combinations (18 total datasets)..."
echo "Split: ${TRAIN_SIZE}-${VAL_SIZE}-${TEST_SIZE} (dev/val/test)"
echo ""

for dataset_config in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse configuration
    IFS='|' read -r SOURCE_DATASET HUB_NAME DISPLAY_NAME ID_STRATEGY REF_DATASET ID_PREFIX <<< "$dataset_config"
    
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$CURRENT/$TOTAL] Processing: $DISPLAY_NAME"
    echo "----------------------------------------------------------------------"
    echo "  Source: $SOURCE_DATASET"
    echo "  Target: $HUB_NAME"
    echo "  ID Strategy: $ID_STRATEGY"
    if [ "$ID_STRATEGY" == "map" ]; then
        echo "  Reference: $REF_DATASET"
    fi
    if [ "$ID_STRATEGY" == "index" ]; then
        echo "  Prefix: $ID_PREFIX"
    fi
    echo ""
    
    # Build command
    CMD="python scripts/split_and_upload_dataset.py \
        --dataset_name \"$SOURCE_DATASET\" \
        --hub_name \"$HUB_NAME\" \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --test_size $TEST_SIZE \
        --seed $SEED \
        --id_strategy \"$ID_STRATEGY\""
        
    if [ -n "$REF_DATASET" ]; then
        CMD="$CMD --reference_dataset \"$REF_DATASET\""
    fi
    
    if [ -n "$ID_PREFIX" ]; then
        CMD="$CMD --id_prefix \"$ID_PREFIX\""
    fi
    
    # Run command
    eval $CMD
    
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

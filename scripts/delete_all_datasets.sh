#!/bin/bash

# Delete all uploaded datasets from wfang11's HuggingFace account
# This will delete all 18 datasets that were uploaded

set -e  # Exit on error

echo "=========================================================================="
echo "Delete All Uploaded Datasets from wfang11's HuggingFace"
echo "=========================================================================="
echo ""

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate weaver environment
echo "Activating weaver conda environment..."
conda activate weaver

# Dataset configurations (same as upload script)
DATASETS=(
    "wfang11/math500-llama70b-1-5-94"
    "wfang11/math500-llama8b-1-5-94"
    "wfang11/gpqa-llama70b-1-5-94"
    "wfang11/gpqa-llama8b-1-5-94"
    "wfang11/mmlu-llama70b-1-5-94"
    "wfang11/mmlu-llama8b-1-5-94"
)

echo "This will delete 18 datasets (6 combinations × 3 splits each):"
echo ""
for dataset in "${DATASETS[@]}"; do
    echo "  - ${dataset}-dev"
    echo "  - ${dataset}-val"
    echo "  - ${dataset}-test"
done
echo ""
echo "⚠️  WARNING: This action cannot be undone!"
echo ""
read -p "Are you sure you want to delete all these datasets? (yes/no) " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

TOTAL=$((${#DATASETS[@]} * 3))
CURRENT=0

echo ""
echo "Starting deletion of $TOTAL datasets..."
echo ""

for dataset in "${DATASETS[@]}"; do
    for split in "dev" "val" "test"; do
        CURRENT=$((CURRENT + 1))
        DATASET_NAME="${dataset}-${split}"
        
        echo "[$CURRENT/$TOTAL] Deleting: $DATASET_NAME"
        
        # Use huggingface-cli to delete the dataset
        if huggingface-cli delete-repo "$DATASET_NAME" --type dataset -y 2>/dev/null; then
            echo "  ✓ Deleted successfully"
        else
            echo "  ⚠️  Failed to delete (may not exist or already deleted)"
        fi
    done
done

echo ""
echo "=========================================================================="
echo "✓ DELETION COMPLETE!"
echo "=========================================================================="
echo ""
echo "All datasets have been processed."
echo "You can now re-upload with the correct seeds using:"
echo "  ./scripts/upload_all_datasets.sh"
echo ""

# Dataset Upload Scripts

Scripts for splitting and uploading HuggingFace datasets.

## Quick Start

Upload all 6 dataset combinations (18 total datasets) to wfang11's HuggingFace:

```bash
./scripts/upload_all_datasets.sh
```

This will create:
- 6 dataset combinations (MATH500, GPQA, MMLU × Llama-70B, Llama-8B)
- Each split into dev (1%), val (5%), test (94%)
- Total: 18 datasets

## What Gets Created

See [DATASET_REFERENCE.md](../DATASET_REFERENCE.md) for the complete list of all 18 datasets.

## Individual Dataset Upload

To upload a single dataset:

```bash
python scripts/split_and_upload_dataset.py \
    --dataset_name hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1 \
    --hub_name wfang11/math500-llama70b-1-5-94 \
    --train_size 0.01 \
    --val_size 0.05 \
    --test_size 0.94
```

## Files

- **`upload_all_datasets.sh`** - Upload all 6 combinations (recommended)
- **`split_and_upload_dataset.py`** - Python script for individual uploads
- **`README.md`** - This file

## Notes

- Splits are deterministic (uses seed=42)
- Running multiple times produces identical splits
- Creates datasets with suffixes: `-dev`, `-val`, `-test`

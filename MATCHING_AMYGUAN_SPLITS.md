# Matching amyguan's Dataset Splits

## Why Were They Different?

Your friend's datasets (`amyguan/math500-k50-1-5-94-*`) differ from yours in two ways:

1. **Random Seeds**:
   - **Your initial uploads**: seed = 42
   - **amyguan's uploads**: seeds = 422, 423

2. **k value (Number of samples)**:
   - **Your uploads**: k = 100 (full samples from hazyresearch)
   - **amyguan's uploads**: k = 50 (subsampled)

## Changes Made to Match Splits

I've updated the scripts to use **seeds 422 and 423** (matching the `verifier_selection/utils.py` logic):

1. **Updated `scripts/upload_all_datasets.sh`**: Changed `SEED=42` to `SEED=422`
2. **Updated `scripts/split_and_upload_dataset.py`**: Now uses two seeds (422, 423) for the two-stage split

## Note on k Value

Your datasets currently have **k=100** (100 samples per problem), while amyguan's have **k=50**.
This means your datasets contain **more data** (twice as many samples per problem).
The train/val/test **splits** (which problems go where) will be identical because of the matching seeds, but the **content** (number of samples) will be different.

## How the Splitting Works

The splitting uses a **two-stage process** (same as `verifier_selection/utils.py`):

```python
# Stage 1: Split off test set (seed 422)
splits = dataset.train_test_split(test_size=0.94, seed=422)
dev_val_ds = splits["train"]  # 6% remaining
test_ds = splits["test"]      # 94%

# Stage 2: Split dev and val (seed 423)
splits = dev_val_ds.train_test_split(test_size=0.05/0.06, seed=423)
dev_ds = splits["train"]  # 1% of original
val_ds = splits["test"]   # 5% of original
```

## Re-uploading to Match

To create datasets that **exactly match** amyguan's splits (in terms of problem assignment):

```bash
./scripts/upload_all_datasets.sh
```

## Verification

You can verify the splits match by comparing:

```python
from datasets import load_dataset

# Load amyguan's dataset
amy_dev = load_dataset('amyguan/math500-k50-1-5-94-dev')['data']

# Load your new dataset
your_dev = load_dataset('wfang11/math500-llama70b-1-5-94-dev')['data']

# Check if they have the same examples (instructions)
print(f"Same first example: {amy_dev[0]['instruction'] == your_dev[0]['instruction']}")

# Check k value
print(f"Amy's k: {len(amy_dev[0]['samples'])}")
print(f"Your k: {len(your_dev[0]['samples'])}")
```

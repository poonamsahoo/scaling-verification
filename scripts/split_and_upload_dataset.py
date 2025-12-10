#!/usr/bin/env python
"""
Split a HuggingFace dataset into train/val/test splits and upload to HuggingFace Hub.

Usage:
    python split_and_upload_dataset.py \
        --dataset_name hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1 \
        --hub_name amyguan/math500-k50-80-10-10 \
        --train_size 0.8 \
        --val_size 0.1 \
        --test_size 0.1 \
        --seed 42 \
        --private
"""

import argparse
import datasets
from datasets import Dataset, load_dataset
import sys


def split_dataset(
    dataset: Dataset,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train/val/test splits.
    
    Args:
        dataset: HuggingFace Dataset to split
        train_size: Proportion for training set (default: 0.8)
        val_size: Proportion for validation set (default: 0.1)
        test_size: Proportion for test set (default: 0.1)
        seed: Random seed for reproducibility (first seed, second seed is seed+1)
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    # Validate split sizes
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")
    
    print(f"\nSplitting dataset with sizes: dev={train_size}, val={val_size}, test={test_size}")
    print(f"Total dataset size: {len(dataset)} examples")
    print(f"Using seeds: {seed}, {seed+1}")
    
    # First split: separate out test set (using first seed)
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    dev_val_ds = splits["train"]
    test_ds = splits["test"]
    
    # Second split: separate dev and val from the remaining data (using second seed)
    # val_size needs to be adjusted relative to the remaining data
    adjusted_val_size = val_size / (train_size + val_size)
    splits = dev_val_ds.train_test_split(test_size=adjusted_val_size, seed=seed + 1)
    dev_ds = splits["train"]
    val_ds = splits["test"]
    
    print(f"\nSplit sizes:")
    print(f"  Dev:   {len(dev_ds)} examples ({len(dev_ds)/len(dataset)*100:.1f}%)")
    print(f"  Val:   {len(val_ds)} examples ({len(val_ds)/len(dataset)*100:.1f}%)")
    print(f"  Test:  {len(test_ds)} examples ({len(test_ds)/len(dataset)*100:.1f}%)")
    
    return dev_ds, val_ds, test_ds

    return dev_ds, val_ds, test_ds


def add_unique_ids(
    dataset: Dataset,
    strategy: str = "none",
    reference_dataset_name: str = None,
    id_prefix: str = None
) -> Dataset:
    """
    Add unique_id column to dataset based on strategy.
    
    Args:
        dataset: HuggingFace Dataset
        strategy: 'none', 'index', or 'map'
        reference_dataset_name: Name of reference dataset for 'map' strategy
        id_prefix: Prefix for 'index' strategy (e.g., 'gpqa')
        
    Returns:
        Dataset with unique_id column
    """
    if "unique_id" in dataset.column_names:
        print(f"Dataset already has 'unique_id' column. Skipping ID generation.")
        return dataset
        
    print(f"\nAdding unique IDs with strategy: {strategy}")
    
    if strategy == "none":
        return dataset
        
    elif strategy == "index":
        if not id_prefix:
            raise ValueError("id_prefix is required for 'index' strategy")
            
        def add_id(example, idx):
            example["unique_id"] = f"{id_prefix}/{idx}"
            return example
            
        return dataset.map(add_id, with_indices=True)
        
    elif strategy == "map":
        if not reference_dataset_name:
            raise ValueError("reference_dataset is required for 'map' strategy")
            
        print(f"Loading reference dataset: {reference_dataset_name}")
        try:
            # Try loading 'data' split first, fall back to 'train'
            try:
                ref_ds = load_dataset(reference_dataset_name, split="data")
            except:
                ref_ds = load_dataset(reference_dataset_name, split="train")
        except Exception as e:
            raise ValueError(f"Could not load reference dataset: {e}")
            
        if "unique_id" not in ref_ds.column_names:
            raise ValueError(f"Reference dataset {reference_dataset_name} does not have 'unique_id' column")
            
        # Create mapping: instruction -> unique_id
        # Normalize instructions by stripping whitespace
        print("Creating instruction -> ID mapping...")
        instr_to_id = {}
        for ex in ref_ds:
            instr = ex.get("instruction", "").strip()
            if instr:
                instr_to_id[instr] = ex["unique_id"]
                
        print(f"Created mapping with {len(instr_to_id)} entries")
        
        def map_id(example):
            instr = example.get("instruction", "").strip()
            if instr in instr_to_id:
                example["unique_id"] = instr_to_id[instr]
            else:
                # Fallback or error? For now, let's error to be safe, or maybe warn?
                # Given we verified the sets match, error is safer to catch issues.
                raise ValueError(f"Could not find mapping for instruction: {instr[:50]}...")
            return example
            
        return dataset.map(map_id)
        
    else:
        raise ValueError(f"Unknown ID strategy: {strategy}")
def upload_splits(
    dev_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    hub_name: str,
    private: bool = False
):
    """
    Upload dataset splits to HuggingFace Hub.
    
    Args:
        dev_ds: Development dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        hub_name: Base name for HuggingFace Hub (e.g., "username/dataset-name")
        private: Whether to make the datasets private
    """
    print(f"\nUploading splits to HuggingFace Hub...")
    print(f"Base name: {hub_name}")
    print(f"Private: {private}")
    
    # Upload dev split (was train)
    dev_hub_name = f"{hub_name}-dev"
    print(f"\nUploading dev split to: {dev_hub_name}")
    dev_ds.push_to_hub(dev_hub_name, private=private)
    print(f"✓ Dev split uploaded successfully")
    
    # Upload val split
    val_hub_name = f"{hub_name}-val"
    print(f"\nUploading val split to: {val_hub_name}")
    val_ds.push_to_hub(val_hub_name, private=private)
    print(f"✓ Val split uploaded successfully")
    
    # Upload test split
    test_hub_name = f"{hub_name}-test"
    print(f"\nUploading test split to: {test_hub_name}")
    test_ds.push_to_hub(test_hub_name, private=private)
    print(f"✓ Test split uploaded successfully")
    
    print(f"\n{'='*60}")
    print(f"All splits uploaded successfully!")
    print(f"{'='*60}")
    print(f"\nYou can now load them with:")
    print(f"  dev_ds = datasets.load_dataset('{dev_hub_name}')['data']")
    print(f"  val_ds = datasets.load_dataset('{val_hub_name}')['data']")
    print(f"  test_ds = datasets.load_dataset('{test_hub_name}')['data']")


def main():
    parser = argparse.ArgumentParser(
        description="Split a HuggingFace dataset and upload to Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split MATH500 into 80-10-10 train/val/test
  python split_and_upload_dataset.py \\
      --dataset_name hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1 \\
      --hub_name amyguan/math500-k50-80-10-10 \\
      --train_size 0.8 --val_size 0.1 --test_size 0.1
  
  # Split with custom sizes (e.g., 1-5-94)
  python split_and_upload_dataset.py \\
      --dataset_name hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1 \\
      --hub_name amyguan/math500-k50-1-5-94 \\
      --train_size 0.01 --val_size 0.05 --test_size 0.94
        """
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name to load (e.g., 'hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1')"
    )
    parser.add_argument(
        "--hub_name",
        type=str,
        required=True,
        help="Base name for uploading to HuggingFace Hub (e.g., 'amyguan/math500-k50-80-10-10'). "
             "Will create {hub_name}-dev, {hub_name}-val, {hub_name}-test"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Proportion of data for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of data for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of data for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded datasets private (default: public)"
    )
    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Skip uploading to HuggingFace Hub (useful for testing)"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="data",
        help="Name of the split to load from the dataset (default: 'data')"
    )
    parser.add_argument(
        "--id_strategy",
        type=str,
        default="none",
        choices=["none", "index", "map"],
        help="Strategy to add unique IDs if missing (default: 'none')"
    )
    parser.add_argument(
        "--reference_dataset",
        type=str,
        help="Reference dataset to map IDs from (required for 'map' strategy)"
    )
    parser.add_argument(
        "--id_prefix",
        type=str,
        help="Prefix for generated IDs (required for 'index' strategy)"
    )
    
    args = parser.parse_args()
    
    # Validate split sizes
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-6:
        print(f"Error: Split sizes must sum to 1.0, got {total}", file=sys.stderr)
        sys.exit(1)
    
    print("="*60)
    print("Dataset Splitting and Upload Script")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Source dataset: {args.dataset_name}")
    print(f"  Target hub name: {args.hub_name}")
    print(f"  Split sizes: train={args.train_size}, val={args.val_size}, test={args.test_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Private: {args.private}")
    print(f"  Upload: {not args.no_upload}")
    
    # Load dataset
    print(f"\nLoading dataset from HuggingFace Hub...")
    try:
        dataset = datasets.load_dataset(args.dataset_name)[args.split_name]
        print(f"✓ Dataset loaded successfully: {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Add unique IDs if requested
    try:
        dataset = add_unique_ids(
            dataset,
            strategy=args.id_strategy,
            reference_dataset_name=args.reference_dataset,
            id_prefix=args.id_prefix
        )
    except Exception as e:
        print(f"Error adding unique IDs: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        dev_ds, val_ds, test_ds = split_dataset(
            dataset,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error splitting dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Upload to HuggingFace Hub
    if not args.no_upload:
        try:
            upload_splits(dev_ds, val_ds, test_ds, args.hub_name, args.private)
        except Exception as e:
            print(f"Error uploading to HuggingFace Hub: {e}", file=sys.stderr)
            print("\nMake sure you're logged in to HuggingFace:")
            print("  huggingface-cli login")
            print("or set your token:")
            print("  export HF_TOKEN='your_token_here'")
            sys.exit(1)
    else:
        print("\n--no_upload flag set, skipping upload to HuggingFace Hub")
        print("\nSplit statistics:")
        print(f"  Dev:   {len(dev_ds)} examples")
        print(f"  Val:   {len(val_ds)} examples")
        print(f"  Test:  {len(test_ds)} examples")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

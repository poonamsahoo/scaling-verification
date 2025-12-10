"""
Preprocessing script to add cluster assignments to val/dev/test splits.

This script:
1. Loads val/dev/test splits from the main dataset
2. Loads cluster assignments from the cluster dataset
3. Matches by problem field
4. Creates 3 separate datasets (val, dev, test) with cluster_id column
5. Saves/uploads to HuggingFace as separate datasets
"""

import argparse
import os
import json
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


def load_main_dataset_splits(base_name: str):
    """Load val, dev, and test splits from the main dataset."""
    print(f"Loading dataset splits from {base_name}...")
    
    val_ds = load_dataset(f"{base_name}-val")["data"]
    dev_ds = load_dataset(f"{base_name}-dev")["data"]
    test_ds = load_dataset(f"{base_name}-test")["data"]
    
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Dev: {len(dev_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    return val_ds, dev_ds, test_ds


def load_cluster_dataset(cluster_column: str, cluster_dataset_path: str):
    """Load cluster assignments from the cluster dataset."""
    print(f"Loading cluster assignments from {cluster_dataset_path}...")
    
    cluster_ds = load_dataset(cluster_dataset_path)
    
    cluster_data = cluster_ds['test']
    
    print(f"  Loaded {len(cluster_data)} cluster assignments")
    
    # Verify cluster column exists
    if cluster_column not in cluster_data.column_names:
        raise ValueError(
            f"Cluster column '{cluster_column}' not found in dataset. "
            f"Available columns: {cluster_data.column_names}"
        )
    
    # Verify problem column exists
    if "problem" not in cluster_data.column_names:
        raise ValueError(
            f"'problem' column not found in cluster dataset. "
            f"Available columns: {cluster_data.column_names}"
        )
    
    print(f"  Using cluster column: {cluster_column}")
    print(f"  Matching by 'problem' field")
    
    # Create a mapping from problem to cluster
    cluster_map = {}
    problems_without_cluster = []
    
    # Build mapping
    for row in tqdm(cluster_data, desc="Building cluster mapping"):
        problem = row.get("problem")
        if problem is not None:
            cluster_id = row[cluster_column]
            # Ensure cluster_id is an integer
            if isinstance(cluster_id, str):
                try:
                    cluster_id = int(cluster_id)
                except ValueError:
                    raise ValueError(f"Cluster ID must be numeric, got: {cluster_id}")
            # Remap cluster IDs to 0-indexed (e.g., 1-5 becomes 0-4)
            cluster_id_0_indexed = int(cluster_id) - 1
            cluster_map[problem] = cluster_id_0_indexed
        else:
            problems_without_cluster.append(row.get("unique_id", "unknown"))
    
    if problems_without_cluster:
        print(f"  ⚠️  Warning: {len(problems_without_cluster)} rows in cluster dataset missing 'problem' field")
    
    print(f"  Created mapping for {len(cluster_map)} problems")
    
    # Check cluster distribution
    cluster_counts = {}
    for cluster_id in cluster_map.values():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    print(f"  Cluster distribution: {dict(sorted(cluster_counts.items()))}")
    
    return cluster_map


def add_clusters_to_split(split_ds, cluster_map: dict, split_name: str):
    """Add cluster assignments to a single split."""
    print(f"\nProcessing {split_name} split...")
    
    # Verify problem column exists in split dataset
    if "problem" not in split_ds.column_names:
        raise ValueError(
            f"'problem' column not found in {split_name} split. "
            f"Available columns: {split_ds.column_names}"
        )
    
    rows = []
    missing_clusters = []
    
    for row in tqdm(split_ds, desc=f"Processing {split_name}"):
        problem = row.get("problem")
        if problem is not None and problem in cluster_map:
            row_dict = dict(row)
            row_dict["cluster_id"] = cluster_map[problem]
            rows.append(row_dict)
        else:
            # Use unique_id or problem for reporting missing clusters
            missing_id = row.get("unique_id", problem)
            missing_clusters.append(missing_id)
    
    if missing_clusters:
        print(f"  ⚠️  Warning: {len(missing_clusters)} problems missing cluster assignments")
        if len(missing_clusters) <= 10:
            print(f"     Missing IDs: {missing_clusters}")
        else:
            print(f"     First 10 missing IDs: {missing_clusters[:10]}")
    
    dataset = Dataset.from_list(rows)
    
    # Check cluster distribution
    cluster_counts = {}
    for row in dataset:
        cluster_id = row.get("cluster_id")
        if cluster_id is not None:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    print(f"  ✅ {split_name}: {len(dataset)} samples")
    print(f"  Cluster distribution: {dict(sorted(cluster_counts.items()))}")
    
    return dataset, missing_clusters


def main():
    parser = argparse.ArgumentParser(
        description="Combine dataset splits with cluster assignments"
    )
    parser.add_argument(
        "--main-dataset-base",
        type=str,
        default="wfang11/math500-llama70b-1-5-94",
        help="Base name for main dataset (will append -val, -dev, -test)"
    )
    parser.add_argument(
        "--cluster-dataset",
        type=str,
        default="pnsahoo/MATH500_LLM_judge_fewshot_meta-llama_Meta-Llama-3-8B-Instruct-Lite",
        help="Path to cluster dataset on HuggingFace"
    )
    parser.add_argument(
        "--cluster-column",
        type=str,
        required=True,
        help="Name of cluster column in cluster dataset"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        required=True,
        help="Base name for output datasets (will create -val, -dev, -test datasets)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace hub (otherwise save locally)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace dataset private (only works with --push-to-hub)"
    )
    
    args = parser.parse_args()
    
    # Load main dataset splits
    val_ds, dev_ds, test_ds = load_main_dataset_splits(args.main_dataset_base)
    
    # Load cluster assignments
    cluster_map = load_cluster_dataset(args.cluster_column, args.cluster_dataset)
    
    # Process each split separately
    val_dataset, val_missing = add_clusters_to_split(val_ds, cluster_map, "val")
    dev_dataset, dev_missing = add_clusters_to_split(dev_ds, cluster_map, "dev")
    test_dataset, test_missing = add_clusters_to_split(test_ds, cluster_map, "test")
    
    total_missing = len(val_missing) + len(dev_missing) + len(test_missing)
    if total_missing > 0:
        print(f"\n⚠️  Total problems missing clusters: {total_missing}")
    
    # Custom cluster remapping: 0 & 1 -> 1, 2->2, 3->3, 4->3
    custom_cluster_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    
    def remap_cluster_id(cluster_id):
        """Remap cluster ID using custom mapping."""
        if cluster_id is None:
            return None
        return custom_cluster_map.get(cluster_id, cluster_id)
    
    def remap_dataset_clusters(dataset):
        """Remap cluster_id values in a dataset."""
        rows = []
        for row in dataset:
            row_dict = dict(row)
            if "cluster_id" in row_dict:
                row_dict["cluster_id"] = remap_cluster_id(row_dict["cluster_id"])
            rows.append(row_dict)
        return Dataset.from_list(rows)
    
    # Remap cluster IDs in all datasets before saving
    print(f"\n🔄 Remapping cluster IDs using mapping: {custom_cluster_map}")
    val_dataset = remap_dataset_clusters(val_dataset)
    dev_dataset = remap_dataset_clusters(dev_dataset)
    test_dataset = remap_dataset_clusters(test_dataset)
    
    # Create DatasetDict for each split (with "data" split as expected by codebase)
    val_dict = DatasetDict({"data": val_dataset})
    dev_dict = DatasetDict({"data": dev_dataset})
    test_dict = DatasetDict({"data": test_dataset})
    
    # Create cluster JSON files for each split
    def create_cluster_json(dataset, split_name):
        cluster_json = {}
        for idx, row in enumerate(dataset):
            cluster_id = row.get("cluster_id")
            if cluster_id is not None:
                # Cluster ID is already remapped in the dataset
                cluster_json[str(idx)] = int(cluster_id)
        return cluster_json
    
    val_json = create_cluster_json(val_dataset, "val")
    dev_json = create_cluster_json(dev_dataset, "dev")
    test_json = create_cluster_json(test_dataset, "test")
    
    # Determine output paths
    if args.push_to_hub:
        val_path = f"{args.output_base}-val"
        dev_path = f"{args.output_base}-dev"
        test_path = f"{args.output_base}-test"
        json_dir = "."
    else:
        val_path = os.path.join(args.output_base, "val")
        dev_path = os.path.join(args.output_base, "dev")
        test_path = os.path.join(args.output_base, "test")
        json_dir = args.output_base
        os.makedirs(args.output_base, exist_ok=True)


    
    
    # Extract dataset name from output_base for JSON file naming
    # If output_base is "user/dataset-name", extract "dataset-name"
    # If output_base is a local path without "/", use the basename
    if "/" in args.output_base:
        dataset_name = args.output_base.split("/", 1)[1]  # Get part after first "/"
    else:
        # For local paths, use the basename
        dataset_name = os.path.basename(args.output_base.rstrip("/"))
    
    # Sanitize dataset name for filename (replace problematic characters)
    dataset_name_safe = dataset_name.replace("/", "_").replace("\\", "_")
    
    # Save cluster JSON files with dataset-specific names
    val_json_path = os.path.join(json_dir, f"cluster_assignments_{dataset_name_safe}_val.json")
    dev_json_path = os.path.join(json_dir, f"cluster_assignments_{dataset_name_safe}_dev.json")
    test_json_path = os.path.join(json_dir, f"cluster_assignments_{dataset_name_safe}_test.json")
    
    with open(val_json_path, 'w') as f:
        json.dump(val_json, f, indent=2)
    with open(dev_json_path, 'w') as f:
        json.dump(dev_json, f, indent=2)
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f"\n💾 Saved cluster JSON files:")
    print(f"   - {val_json_path}")
    print(f"   - {dev_json_path}")
    print(f"   - {test_json_path}")
    
    # Save or push datasets
    if args.push_to_hub:
        print(f"\n📤 Pushing to HuggingFace...")
        val_dict.push_to_hub(val_path, private=args.private)
        print(f"   ✅ {val_path}")
        dev_dict.push_to_hub(dev_path, private=args.private)
        print(f"   ✅ {dev_path}")
        test_dict.push_to_hub(test_path, private=args.private)
        print(f"   ✅ {test_path}")
    else:
        print(f"\n💾 Saving locally...")
        val_dict.save_to_disk(val_path)
        print(f"   ✅ {val_path}")
        dev_dict.save_to_disk(dev_path)
        print(f"   ✅ {dev_path}")
        test_dict.save_to_disk(test_path)
        print(f"   ✅ {test_path}")
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nCreated 3 separate datasets:")
    print(f"  - Val: {val_path} ({len(val_dataset)} samples)")
    print(f"  - Dev: {dev_path} ({len(dev_dataset)} samples)")
    print(f"  - Test: {test_path} ({len(test_dataset)} samples)")
    # Calculate remapped cluster count
    remapped_cluster_ids = set()
    for dataset in [val_dataset, dev_dataset, test_dataset]:
        for row in dataset:
            cluster_id = row.get("cluster_id")
            if cluster_id is not None:
                remapped_cluster_ids.add(cluster_id)
    n_clusters = len(remapped_cluster_ids)
    
    print(f"\nTo use in your config files:")
    print(f"\nFor TRAINING (use dev split):")
    print(f'  data_cfg.dataset_path: "{dev_path}"')
    print(f'  data_cfg.train_split: 1.0  # Use all dev data for training')
    print(f'  model_cfg.cluster_cfg.cluster_type: "json"')
    print(f'  model_cfg.cluster_cfg.embedding_model: "{os.path.abspath(dev_json_path)}"')
    print(f'  model_cfg.cluster_cfg.n_clusters: {n_clusters}')
    print(f"\nFor EVALUATION (use test split):")
    print(f'  data_cfg.dataset_path: "{test_path}"')
    print(f'  data_cfg.train_split: 0.0  # Use all test data for evaluation')
    print(f'  model_cfg.cluster_cfg.cluster_type: "json"')
    print(f'  model_cfg.cluster_cfg.embedding_model: "{os.path.abspath(test_json_path)}"')
    print(f'  model_cfg.cluster_cfg.n_clusters: {n_clusters}')


if __name__ == "__main__":
    main()


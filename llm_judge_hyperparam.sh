#!/usr/bin/env bash
set -euo pipefail

# Cluster-based hyperparameter search script
# This script processes a CSV with cluster-specific verifier selections and hyperparameters.
# For each row in the CSV (which includes cluster_id, k, alpha, beta, gamma, selected_verifiers),
# it runs weaver training/evaluation with cluster-specific verifier subsets.
# The cluster_id is included in the run name for tracking.
#
# Usage:
#   bash llm_judge_hyperparam.sh --config-name subset


CSV_PATH="/Users/poonamsahoo/scaling-verification/results/GPQA-llama8b-qwen-judge-amended-verifier_hparam_search_results_10percent.csv"
DATASET_PATH="pnsahoo/GPQA-llama8b-qwen-judge-dev"
CONFIG_NAME="subset"
LOG_DATASET_NAME="GPQA-llama8b-qwen-judge-amended"
LOG_DATASET_DEV="dev"
VERIFIER_COL="selected_verifiers"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      CSV_PATH="$2"; shift 2 ;;
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --config-name)
      CONFIG_NAME="$2"; shift 2 ;;
    --log-name)
      LOG_DATASET_NAME="$2"; shift 2 ;;
    --log-dev)
      LOG_DATASET_DEV="$2"; shift 2 ;;
    --verifier-col)
      VERIFIER_COL="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "--dataset-path is required (e.g., amyguan/GPQA-k50-80-10-10-val)" >&2
  exit 1
fi

if [[ -z "$LOG_DATASET_NAME" ]]; then
  echo "--log-name is required (e.g., GPQA-k50-80-10-10)" >&2
  exit 1
fi

# Create temporary directory for filtered datasets
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT
echo "Using temporary directory for filtered datasets: $TEMP_DIR"

# Generate Hydra list strings from CSV column and loop
# Now includes cluster_id for cluster-based hyperparameter search with dataset filtering
while IFS=$'\t' read -r HYDRA_LIST NAME_SUFFIX CLUSTER_ID K A B G FILTERED_DATASET_PATH; do
  [[ -n "$NAME_SUFFIX" ]] && NAME_SUFFIX="-$NAME_SUFFIX" || NAME_SUFFIX=""
  # Include cluster_id in the run name
  if [[ -n "$CLUSTER_ID" && "$CLUSTER_ID" != "None" && "$CLUSTER_ID" != "" ]]; then
    RUN_LOG_NAME="${LOG_DATASET_NAME}-cluster${CLUSTER_ID}${NAME_SUFFIX}"
  else
    RUN_LOG_NAME="${LOG_DATASET_NAME}${NAME_SUFFIX}"
  fi
  echo "Running cluster ${CLUSTER_ID} with verifier subset: ${HYDRA_LIST} | name suffix: ${NAME_SUFFIX}"
  echo "Using filtered dataset: ${FILTERED_DATASET_PATH}"
  
  # Use subset config with per_dataset model_class (not cluster)
  python selection/run.py \
    --config-name "$CONFIG_NAME" \
    data_cfg.dataset_path="$FILTERED_DATASET_PATH" \
    model_cfg.model_class="per_dataset" \
    verifier_cfg.verifier_subset="$HYDRA_LIST" \
    k="$K" alpha="$A" beta="$B" gamma="$G" \
    log_dataset_name="$RUN_LOG_NAME" \
    log_dataset_dev="$LOG_DATASET_DEV"
done < <(python - "$CSV_PATH" "$VERIFIER_COL" "$DATASET_PATH" "$TEMP_DIR" <<'PY'
import sys
import os
import pandas as pd
from datasets import load_dataset, Dataset

csv_path = sys.argv[1]
col = sys.argv[2]
dataset_path = sys.argv[3]
temp_dir = sys.argv[4]

df = pd.read_csv(csv_path)

if col not in df.columns:
    raise SystemExit(f"Column '{col}' not found in {csv_path}. Available: {list(df.columns)}")

# Check if cluster_id column exists
if "cluster_id" not in df.columns:
    raise SystemExit(f"Column 'cluster_id' not found in {csv_path}. This script requires cluster-based hyperparameter search.")

def fmt(x):
    try:
        return format(float(x), 'g')
    except Exception:
        return str(x)

# Cache filtered datasets by cluster_id to avoid reloading
filtered_datasets = {}

for _, row in df.iterrows():
    val = str(row[col])
    # Expect comma-separated names, possibly with spaces
    # e.g., "A, B, C" -> ['A','B','C']
    names = [s.strip() for s in val.split(',') if s.strip()]
    hydra_list = "[" + ",".join("'" + n + "'" for n in names) + "]"
    
    # Get cluster_id
    cluster_id = row.get('cluster_id', None)
    if cluster_id is None or pd.isna(cluster_id):
        raise SystemExit(f"Invalid cluster_id in row: {row.to_dict()}")
    cluster_id = int(cluster_id)
    cluster_id_str = str(cluster_id)
    
    # Create filtered dataset for this cluster if not already created
    if cluster_id not in filtered_datasets:
        print(f"Filtering dataset {dataset_path} to cluster_id={cluster_id}...", file=sys.stderr)
        try:
            # Load the dataset
            if os.path.exists(dataset_path):
                from datasets import load_from_disk
                dataset = load_from_disk(dataset_path)
            else:
                dataset = load_dataset(dataset_path)
            
            # Get the data split (usually "data" or the split name)
            if isinstance(dataset, dict):
                if "data" in dataset:
                    data = dataset["data"]
                else:
                    # Use the first split
                    data = dataset[list(dataset.keys())[0]]
            else:
                data = dataset
            
            # Filter by cluster_id
            if "cluster_id" not in data.column_names:
                raise SystemExit(f"cluster_id column not found in dataset {dataset_path}")
            
            filtered_data = data.filter(lambda x: x["cluster_id"] == cluster_id)
            
            if len(filtered_data) == 0:
                print(f"Warning: No samples found for cluster_id {cluster_id}, skipping...", file=sys.stderr)
                continue
            
            # Save filtered dataset to temp directory
            # Save as Dataset (not DatasetDict) since weaver code expects Dataset when loading from disk
            filtered_path = os.path.join(temp_dir, f"cluster_{cluster_id}")
            filtered_data.save_to_disk(filtered_path)
            filtered_datasets[cluster_id] = filtered_path
            print(f"Saved filtered dataset to {filtered_path} ({len(filtered_data)} samples)", file=sys.stderr)
        except Exception as e:
            raise SystemExit(f"Error filtering dataset for cluster_id {cluster_id}: {e}")
    
    filtered_dataset_path = filtered_datasets[cluster_id]
    
    # Build a run-name suffix using k, alpha, beta, gamma columns if present
    k = row.get('k', None)
    a = row.get('alpha', None)
    b = row.get('beta', None)
    g = row.get('gamma', None)
    parts = []
    if k is not None: parts.append(f"k{fmt(k)}")
    if a is not None: parts.append(f"a{fmt(a)}")
    if b is not None: parts.append(f"b{fmt(b)}")
    if g is not None: parts.append(f"g{fmt(g)}")
    suffix = "-".join(parts) if parts else ""
    print("\t".join([
        hydra_list,
        suffix,
        cluster_id_str,
        str(row.get('k', '')),
        str(row.get('alpha', '')),
        str(row.get('beta', '')),
        str(row.get('gamma', '')),
        filtered_dataset_path,
    ]))
PY)


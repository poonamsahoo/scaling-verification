#!/usr/bin/env bash
set -euo pipefail

# Final evaluation script for LLM judge verifiers
# For each cluster:
#   1. Get the best hyperparameters + verifiers from dev set results (CSV)
#   2. Train weaver on TEST set using those hyperparameters
#   3. Report per-cluster and overall performance
#
# Usage:
#   bash weaver/final_eval_llm_judge.sh \
#     --results-dir results \
#     --dataset-path pnsahoo/math500-llama8b-qwen-judge-amended-test \
#     --config-name subset \
#     --log-name math500-llama8b-qwen-judge-amended \
#     --log-dev test

RESULTS_DIR="results"
DATASET_PATH=""
CONFIG_NAME="subset"
LOG_DATASET_NAME="GPQA-llama8b-qwen-judge-amended"
LOG_DATASET_DEV="test"
CSV_PREFIX="hparam_search_summary_GPQA-llama8b-qwen-judge-amended"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR="$2"; shift 2 ;;
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --config-name)
      CONFIG_NAME="$2"; shift 2 ;;
    --log-name)
      LOG_DATASET_NAME="$2"; shift 2 ;;
    --log-dev)
      LOG_DATASET_DEV="$2"; shift 2 ;;
    --csv-prefix)
      CSV_PREFIX="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "--dataset-path is required (e.g., pnsahoo/math500-llama8b-qwen-judge-amended-test)" >&2
  exit 1
fi

if [[ -z "$LOG_DATASET_NAME" ]]; then
  echo "--log-name is required (e.g., math500-llama8b-qwen-judge-amended)" >&2
  exit 1
fi

echo "=========================================="
echo "Final Evaluation Script"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Dataset path: $DATASET_PATH"
echo "Config name: $CONFIG_NAME"
echo "Log dataset name: $LOG_DATASET_NAME"
echo "Log dataset dev: $LOG_DATASET_DEV"
echo "CSV prefix: $CSV_PREFIX"
echo "=========================================="

# Create temporary directory for filtered datasets
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT
echo "Using temporary directory for filtered datasets: $TEMP_DIR"

# Find all cluster CSV files
CLUSTER_CSV_FILES=($(ls "$RESULTS_DIR/${CSV_PREFIX}-cluster"*"_dev.csv" 2>/dev/null | sort -V))

if [[ ${#CLUSTER_CSV_FILES[@]} -eq 0 ]]; then
  echo "Error: No cluster CSV files found matching pattern: $RESULTS_DIR/${CSV_PREFIX}-cluster*_dev.csv" >&2
  exit 1
fi

echo "Found ${#CLUSTER_CSV_FILES[@]} cluster CSV files:"
for csv in "${CLUSTER_CSV_FILES[@]}"; do
  echo "  - $(basename $csv)"
done
echo ""

# Process each cluster
declare -a CLUSTER_RESULTS_SELECT
declare -a CLUSTER_RESULTS_SAMPLE
declare -a CLUSTER_IDS

for CSV_FILE in "${CLUSTER_CSV_FILES[@]}"; do
  # Extract cluster ID from filename (e.g., cluster0, cluster1, etc.)
  CLUSTER_ID=$(basename "$CSV_FILE" | sed -n 's/.*cluster\([0-9]*\)_dev\.csv/\1/p')
  
  if [[ -z "$CLUSTER_ID" ]]; then
    echo "Warning: Could not extract cluster ID from $CSV_FILE, skipping..." >&2
    continue
  fi
  
  echo "=========================================="
  echo "Processing Cluster $CLUSTER_ID"
  echo "=========================================="
  
  # Get best row from CSV (sorted by test_select_accuracy, test_sample_accuracy, then num_verifiers)
  BEST_ROW=$(python - "$CSV_FILE" <<'PY'
import sys
import pandas as pd
import numpy as np

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# Check required columns
if 'test_select_accuracy' not in df.columns:
    raise SystemExit(f"Column 'test_select_accuracy' not found in {csv_path}")

# Prepare sorting columns
sort_cols = ['test_select_accuracy']
ascending = [False]

# Add test_sample_accuracy if available
if 'test_sample_accuracy' in df.columns:
    sort_cols.append('test_sample_accuracy')
    ascending.append(False)
else:
    # If missing, add a dummy column with NaN (will be sorted last)
    df['test_sample_accuracy'] = np.nan
    sort_cols.append('test_sample_accuracy')
    ascending.append(False)

# Add num_verifiers as tiebreaker (ascending - prefer fewer verifiers)
if 'num_verifiers' in df.columns:
    sort_cols.append('num_verifiers')
    ascending.append(True)
else:
    # If missing, try to compute from verifiers column
    if 'verifiers' in df.columns:
        try:
            import ast
            df['num_verifiers'] = df['verifiers'].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
            )
            sort_cols.append('num_verifiers')
            ascending.append(True)
        except:
            # If we can't compute, add dummy column
            df['num_verifiers'] = np.inf
            sort_cols.append('num_verifiers')
            ascending.append(True)
    else:
        # If verifiers column also missing, add dummy
        df['num_verifiers'] = np.inf
        sort_cols.append('num_verifiers')
        ascending.append(True)

# Sort and get best row
df_sorted = df.sort_values(by=sort_cols, ascending=ascending, na_position='last')
best_row = df_sorted.iloc[0]

# Extract verifiers (convert from string representation of list to actual list)
import ast
verifiers_str = best_row['verifiers']
try:
    verifiers_list = ast.literal_eval(verifiers_str)
except:
    # Fallback: try to parse as string
    verifiers_list = [v.strip().strip("'\"") for v in verifiers_str.strip('[]').split(',')]

# Format as Hydra list
hydra_list = "[" + ",".join("'" + str(v) + "'" for v in verifiers_list) + "]"

# Extract hyperparameters
k = best_row.get('k', '')
alpha = best_row.get('alpha', '')
beta = best_row.get('beta', '')
gamma = best_row.get('gamma', '')
test_select_acc = best_row['test_select_accuracy']
test_sample_acc = best_row.get('test_sample_accuracy', 'nan')
num_verifiers = best_row.get('num_verifiers', 'nan')

print(f"{hydra_list}\t{k}\t{alpha}\t{beta}\t{gamma}\t{test_select_acc}\t{test_sample_acc}\t{num_verifiers}")
PY
)
  
  if [[ -z "$BEST_ROW" ]]; then
    echo "Error: Could not find best row in $CSV_FILE" >&2
    continue
  fi
  
  IFS=$'\t' read -r HYDRA_LIST K A B G DEV_SELECT_ACC DEV_SAMPLE_ACC NUM_VERIFIERS <<< "$BEST_ROW"
  
  echo "Best dev performance:"
  echo "  test_select_accuracy = $DEV_SELECT_ACC"
  echo "  test_sample_accuracy = $DEV_SAMPLE_ACC"
  echo "  num_verifiers = $NUM_VERIFIERS"
  echo "Hyperparameters: k=$K, alpha=$A, beta=$B, gamma=$G"
  echo "Verifiers: ${HYDRA_LIST:0:100}..."
  echo ""
  
  # Filter dataset to this cluster
  echo "Filtering dataset to cluster $CLUSTER_ID..."
  FILTERED_DATASET_PATH=$(python - "$DATASET_PATH" "$CLUSTER_ID" "$TEMP_DIR" <<'PY'
import sys
import os
from datasets import load_dataset, load_from_disk, Dataset

dataset_path = sys.argv[1]
cluster_id = int(sys.argv[2])
temp_dir = sys.argv[3]

# Load the dataset
if os.path.exists(dataset_path):
    from datasets import load_from_disk
    dataset = load_from_disk(dataset_path)
else:
    dataset = load_dataset(dataset_path)

# Get the data split
if isinstance(dataset, dict):
    if "data" in dataset:
        data = dataset["data"]
    else:
        data = dataset[list(dataset.keys())[0]]
else:
    data = dataset

# Filter by cluster_id
if "cluster_id" not in data.column_names:
    raise SystemExit(f"cluster_id column not found in dataset {dataset_path}")

filtered_data = data.filter(lambda x: x["cluster_id"] == cluster_id)

if len(filtered_data) == 0:
    raise SystemExit(f"No samples found for cluster_id {cluster_id}")

# Save filtered dataset
filtered_path = os.path.join(temp_dir, f"cluster_{cluster_id}")
filtered_data.save_to_disk(filtered_path)
print(filtered_path)
PY
)
  
  if [[ -z "$FILTERED_DATASET_PATH" ]]; then
    echo "Error: Could not create filtered dataset for cluster $CLUSTER_ID" >&2
    continue
  fi
  
  echo "Filtered dataset saved to: $FILTERED_DATASET_PATH"
  echo ""
  
  # Run weaver on test set
  RUN_LOG_NAME="${LOG_DATASET_NAME}-cluster${CLUSTER_ID}-final"
  echo "Running weaver on test set..."
  echo "Run name: $RUN_LOG_NAME"
  echo ""
  
  # Build the command with hyperparameters
  CMD_ARGS=(
    "--config-name" "$CONFIG_NAME"
    "data_cfg.dataset_path=$FILTERED_DATASET_PATH"
    "model_cfg.model_class=per_dataset"
    "verifier_cfg.verifier_type=specific_subset"
    "verifier_cfg.verifier_subset=$HYDRA_LIST"
    "log_dataset_name=$RUN_LOG_NAME"
    "log_dataset_dev=$LOG_DATASET_DEV"
  )
  
  # Add hyperparameters if they're not empty
  [[ -n "$K" && "$K" != "nan" && "$K" != "None" ]] && CMD_ARGS+=("k=$K")
  [[ -n "$A" && "$A" != "nan" && "$A" != "None" ]] && CMD_ARGS+=("alpha=$A")
  [[ -n "$B" && "$B" != "nan" && "$B" != "None" ]] && CMD_ARGS+=("beta=$B")
  [[ -n "$G" && "$G" != "nan" && "$G" != "None" ]] && CMD_ARGS+=("gamma=$G")
  
  # Run the command
  python selection/run.py "${CMD_ARGS[@]}"
  
  # Extract results from the output (we'll parse from the summary CSV that gets created)
  # The run.py script creates a summary CSV in results/ directory
  SUMMARY_FILE="results/hparam_search_summary_${RUN_LOG_NAME}_${LOG_DATASET_DEV}.csv"
  
  if [[ -f "$SUMMARY_FILE" ]]; then
    # Get both test_select_accuracy and test_sample_accuracy from the summary
    TEST_RESULTS=$(python - "$SUMMARY_FILE" <<'PY'
import sys
import pandas as pd

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
if len(df) > 0:
    last_row = df.iloc[-1]
    select_acc = last_row.get('test_select_accuracy', 'nan')
    sample_acc = last_row.get('test_sample_accuracy', 'nan')
    print(f"{select_acc}\t{sample_acc}")
else:
    print("nan\tnan")
PY
)
    IFS=$'\t' read -r TEST_SELECT_ACC TEST_SAMPLE_ACC <<< "$TEST_RESULTS"
    CLUSTER_RESULTS_SELECT+=("$TEST_SELECT_ACC")
    CLUSTER_RESULTS_SAMPLE+=("$TEST_SAMPLE_ACC")
    CLUSTER_IDS+=("$CLUSTER_ID")
    echo "Cluster $CLUSTER_ID results:"
    echo "  test_select_accuracy: $TEST_SELECT_ACC"
    echo "  test_sample_accuracy: $TEST_SAMPLE_ACC"
  else
    echo "Warning: Summary file not found: $SUMMARY_FILE"
    CLUSTER_RESULTS_SELECT+=("nan")
    CLUSTER_RESULTS_SAMPLE+=("nan")
    CLUSTER_IDS+=("$CLUSTER_ID")
  fi
  
  echo ""
done

# Report overall results
echo "=========================================="
echo "FINAL RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "Per-Cluster Performance:"
echo "------------------------"

# Calculate weighted averages (if we have problem counts)
TOTAL_SELECT_ACC=0.0
TOTAL_SAMPLE_ACC=0.0
TOTAL_PROBLEMS=0
VALID_CLUSTERS=0
HAS_SAMPLE_ACC=false

for i in "${!CLUSTER_IDS[@]}"; do
  CLUSTER_ID="${CLUSTER_IDS[$i]}"
  SELECT_ACC="${CLUSTER_RESULTS_SELECT[$i]}"
  SAMPLE_ACC="${CLUSTER_RESULTS_SAMPLE[$i]}"
  
  if [[ "$SELECT_ACC" != "nan" && -n "$SELECT_ACC" ]]; then
    echo "  Cluster $CLUSTER_ID:"
    echo "    test_select_accuracy: $SELECT_ACC"
    if [[ "$SAMPLE_ACC" != "nan" && -n "$SAMPLE_ACC" ]]; then
      echo "    test_sample_accuracy: $SAMPLE_ACC"
    else
      echo "    test_sample_accuracy: N/A"
    fi
    # Try to get problem count and num_verifiers from the summary file
    SUMMARY_FILE="results/hparam_search_summary_${LOG_DATASET_NAME}-cluster${CLUSTER_ID}-final_${LOG_DATASET_DEV}.csv"
    if [[ -f "$SUMMARY_FILE" ]]; then
      SUMMARY_INFO=$(python - "$SUMMARY_FILE" <<'PY'
import sys
import pandas as pd
try:
    df = pd.read_csv(sys.argv[1])
    if len(df) > 0:
        last_row = df.iloc[-1]
        num_problems = last_row.get('test_problems', 1)
        num_verifiers = last_row.get('num_verifiers', 'nan')
        print(f"{num_problems}\t{num_verifiers}")
    else:
        print("1\tnan")
except:
    print("1\tnan")
PY
)
      IFS=$'\t' read -r NUM_PROBLEMS NUM_VERIFIERS <<< "$SUMMARY_INFO"
      
      if [[ "$NUM_VERIFIERS" != "nan" && -n "$NUM_VERIFIERS" ]]; then
        echo "    num_verifiers: $NUM_VERIFIERS"
      fi
      TOTAL_SELECT_ACC=$(python -c "print($TOTAL_SELECT_ACC + $SELECT_ACC * $NUM_PROBLEMS)")
      if [[ "$SAMPLE_ACC" != "nan" && -n "$SAMPLE_ACC" ]]; then
        TOTAL_SAMPLE_ACC=$(python -c "print($TOTAL_SAMPLE_ACC + $SAMPLE_ACC * $NUM_PROBLEMS)")
        HAS_SAMPLE_ACC=true
      fi
      TOTAL_PROBLEMS=$(python -c "print($TOTAL_PROBLEMS + $NUM_PROBLEMS)")
      VALID_CLUSTERS=$((VALID_CLUSTERS + 1))
    else
      # Fallback: equal weight
      TOTAL_SELECT_ACC=$(python -c "print($TOTAL_SELECT_ACC + $SELECT_ACC)")
      if [[ "$SAMPLE_ACC" != "nan" && -n "$SAMPLE_ACC" ]]; then
        TOTAL_SAMPLE_ACC=$(python -c "print($TOTAL_SAMPLE_ACC + $SAMPLE_ACC)")
        HAS_SAMPLE_ACC=true
      fi
      TOTAL_PROBLEMS=$(python -c "print($TOTAL_PROBLEMS + 1)")
      VALID_CLUSTERS=$((VALID_CLUSTERS + 1))
    fi
  else
    echo "  Cluster $CLUSTER_ID: N/A (no valid results)"
  fi
done

echo ""
if [[ $VALID_CLUSTERS -gt 0 && $TOTAL_PROBLEMS -gt 0 ]]; then
  OVERALL_SELECT_ACC=$(python -c "print($TOTAL_SELECT_ACC / $TOTAL_PROBLEMS)")
  echo "Overall Performance (weighted average):"
  echo "  test_select_accuracy: $OVERALL_SELECT_ACC"
  if [[ "$HAS_SAMPLE_ACC" == "true" ]]; then
    OVERALL_SAMPLE_ACC=$(python -c "print($TOTAL_SAMPLE_ACC / $TOTAL_PROBLEMS)")
    echo "  test_sample_accuracy: $OVERALL_SAMPLE_ACC"
  else
    echo "  test_sample_accuracy: N/A"
  fi
  echo "Total problems: $TOTAL_PROBLEMS"
  echo "Valid clusters: $VALID_CLUSTERS"
else
  echo "Overall Performance: N/A (no valid results)"
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="

#!/usr/bin/env bash
set -euo pipefail

# bash /Users/amyguan/Documents/cs329a/scaling-verification/hyperparam.sh \
#   --csv /Users/amyguan/Documents/cs329a/scaling-verification/verifier_selection/verifier_hparam_search_results.csv \
#   --dataset-path amyguan/math500-k50-80-10-10-val \
#   --config-name subset \
#   --log-name math500-k50-80-10-10 \
#   --log-dev val

# bash /Users/amyguan/Documents/cs329a/scaling-verification/hyperparam.sh \
#   --csv /Users/amyguan/Documents/cs329a/scaling-verification/verifier_selection/verifier_hparam_search_results.csv \
#   --dataset-path amyguan/math500-k50-80-10-10-dev \
#   --config-name subset \
#   --log-name math500-k50-80-10-10 \
#   --log-dev dev

CSV_PATH="/Users/amyguan/Documents/cs329a/scaling-verification/results/verifier_hparam_search_results.csv"
DATASET_PATH="amyguan/math500-k50-80-10-10-val"
CONFIG_NAME="subset"
LOG_DATASET_NAME="math500-k50-80-10-10"
LOG_DATASET_DEV="dev"
VERIFIER_COL="selected_verifiers"   # override with --verifier-col if your CSV differs

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
  echo "--dataset-path is required (e.g., amyguan/math500-k50-80-10-10-val)" >&2
  exit 1
fi

if [[ -z "$LOG_DATASET_NAME" ]]; then
  echo "--log-name is required (e.g., math500-k50-80-10-10)" >&2
  exit 1
fi

# Generate Hydra list strings from CSV column and loop
while IFS=$'\t' read -r HYDRA_LIST NAME_SUFFIX K A B G; do
  [[ -n "$NAME_SUFFIX" ]] && NAME_SUFFIX="-$NAME_SUFFIX" || NAME_SUFFIX=""
  RUN_LOG_NAME="${LOG_DATASET_NAME}${NAME_SUFFIX}"
  echo "Running with verifier subset: ${HYDRA_LIST} | name suffix: ${NAME_SUFFIX}"
  python selection/run.py \
    --config-name "$CONFIG_NAME" \
    data_cfg.dataset_path="$DATASET_PATH" \
    verifier_cfg.verifier_subset="$HYDRA_LIST" \
    k="$K" alpha="$A" beta="$B" gamma="$G" \
    log_dataset_name="$RUN_LOG_NAME" \
    log_dataset_dev="$LOG_DATASET_DEV"
done < <(python - "$CSV_PATH" "$VERIFIER_COL" <<'PY'
import sys
import pandas as pd

csv_path = sys.argv[1]
col = sys.argv[2]
df = pd.read_csv(csv_path)

if col not in df.columns:
    raise SystemExit(f"Column '{col}' not found in {csv_path}. Available: {list(df.columns)}")

def fmt(x):
    try:
        return format(float(x), 'g')
    except Exception:
        return str(x)

for _, row in df.iterrows():
    val = str(row[col])
    # Expect comma-separated names, possibly with spaces
    # e.g., "A, B, C" -> ['A','B','C']
    names = [s.strip() for s in val.split(',') if s.strip()]
    hydra_list = "[" + ",".join("'" + n + "'" for n in names) + "]"
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
        str(row.get('k', '')),
        str(row.get('alpha', '')),
        str(row.get('beta', '')),
        str(row.get('gamma', '')),
    ]))
PY)
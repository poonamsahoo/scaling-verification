# Multi-Dataset Grid Search Guide

This guide explains how to run comprehensive grid search across all 18 datasets with verifier selection and Weaver evaluation.

## Quick Start

```bash
# 1. Test on a single dataset first (recommended)
python test_single_dataset.py \
    --dataset wfang11/math500-llama70b-1-5-94-dev \
    --name math500-llama70b-test

# 2. Run full grid search on all 18 datasets
python run_all_datasets_grid_search.py --config datasets_config.yaml

# 3. Aggregate results
python aggregate_results.py --results_dir results

# 4. Generate report
python generate_report.py --results_dir results
```

## Overview

The grid search pipeline consists of:

1. **Stage 1: Verifier Selection** - Use greedy algorithm to select optimal verifier subsets
2. **Stage 2: Weaver Evaluation** - Evaluate Weaver with selected verifiers
3. **Aggregation** - Combine results from all datasets
4. **Reporting** - Generate analysis report with visualizations

## File Structure

```
.
├── datasets_config.yaml                    # Configuration for all 18 datasets
├── test_single_dataset.py                  # Test script (run this first!)
├── run_all_datasets_grid_search.py         # Master orchestration script
├── aggregate_results.py                    # Results aggregation
├── generate_report.py                      # Report generation
│
├── verifier_selection/
│   ├── run_verifier_selection.py          # Stage 1 (refactored)
│   ├── utils.py                            # Utility functions
│   └── search.py                           # Original script (for reference)
│
├── selection/
│   ├── run_weaver_evaluation.py           # Stage 2 (wrapper)
│   └── run.py                              # Original Weaver script
│
└── results/                                # Output directory
    ├── all_datasets_summary.csv            # Combined results
    ├── execution_log.txt                   # Execution log
    ├── report.md                           # Final report
    ├── report_figures/                     # Visualizations
    │
    ├── math500-llama70b/                   # Per-dataset results
    │   ├── verifier_selection_results.csv
    │   ├── weaver_train_*.csv
    │   ├── weaver_test_*.csv
    │   └── plots/
    │
    ├── math500-llama8b/
    ├── gpqa-llama70b/
    └── ...                                 # All 18 datasets
```

## Configuration

### datasets_config.yaml

Main configuration file defining:

```yaml
datasets:
  - name: "math500-llama70b"
    dev: "wfang11/math500-llama70b-1-5-94-dev"
    # ... (all 18 datasets)

hyperparameters:
  k: [5, 10, 15]              # Number of verifiers
  alpha: [1.0]                # Utility weight (fixed)
  beta: [0.25, 0.5, 1.0]      # Similarity penalty
  gamma: [0.25, 0.5, 1.0]     # Cost penalty

wandb:
  entity: "329a"
  project: "verification"
  enabled: true
```

**Total configurations per dataset:** 3 (k) × 1 (alpha) × 3 (beta) × 3 (gamma) = **27 configurations**

**Total configurations across all datasets:** 27 × 18 = **486 configurations**

## Step-by-Step Usage

### Step 1: Test Pipeline (IMPORTANT!)

Before running on all 18 datasets, test on a single dataset:

```bash
# Test with MATH500 70B (recommended)
python test_single_dataset.py \
    --dataset wfang11/math500-llama70b-1-5-94-dev \
    --name math500-llama70b-test \
    --output_dir test_results

# Or test with GPQA 8B
python test_single_dataset.py \
    --dataset wfang11/gpqa-llama8b-1-5-94-dev \
    --name gpqa-llama8b-test
```

**What the test does:**
- Runs verifier selection with reduced hyperparameter space (8 configs instead of 27)
- Evaluates Weaver on best configuration
- Saves results to `test_results/`
- Verifies everything works before full run

**Expected time:** ~10-20 minutes

### Step 2: Run Full Grid Search

Once test succeeds, run on all datasets:

```bash
# Run all 18 datasets
python run_all_datasets_grid_search.py --config datasets_config.yaml

# Or run specific datasets only
python run_all_datasets_grid_search.py --config datasets_config.yaml \
    --datasets math500-llama70b gpqa-llama70b mmlu-llama70b

# Resume from interrupted run
python run_all_datasets_grid_search.py --config datasets_config.yaml
```

**Features:**
- **Progress tracking** - Shows progress bar
- **Resume capability** - Automatically skips completed datasets
- **Error handling** - Continues even if one dataset fails
- **Logging** - Detailed log in `results/execution_log.txt`

**Expected time:** ~9-18 hours for all 18 datasets (sequential)

**Run in background:**
```bash
# Using screen
screen -S grid_search
python run_all_datasets_grid_search.py --config datasets_config.yaml
# Press Ctrl+A, D to detach
# screen -r grid_search to reattach

# Using nohup
nohup python run_all_datasets_grid_search.py --config datasets_config.yaml > grid_search.log 2>&1 &
```

### Step 3: Aggregate Results

After grid search completes:

```bash
python aggregate_results.py --results_dir results
```

**Creates:**
- `verifier_selection_all_datasets.csv` - All Stage 1 results
- `weaver_evaluation_all_datasets.csv` - All Stage 2 results
- `best_configs_per_dataset.csv` - Best config for each dataset
- `verifier_frequency_analysis.csv` - Which verifiers selected most
- `summary_statistics.json` - Overall statistics

### Step 4: Generate Report

```bash
python generate_report.py --results_dir results --output results/report.md
```

**Creates:**
- `report.md` - Comprehensive markdown report
- `report_figures/` - Visualizations:
  - Verifier frequency bar chart
  - Accuracy by dataset type/model size
  - Hyperparameter heatmaps
  - Dataset comparison plots

**View report:**
```bash
cat results/report.md
# Or open in editor/browser
```

## Output Files Explained

### Per-Dataset Results

Each dataset gets a subdirectory with:

```
results/math500-llama70b/
├── verifier_selection_results.csv       # All 27 configs from Stage 1
├── weaver_all_configs_summary.csv       # Summary of all 27 Weaver runs
├── weaver_train_k10_b0.5_g0.5.csv      # Detailed train results (per config)
├── weaver_test_k10_b0.5_g0.5.csv       # Detailed test results (per config)
├── weaver_summary_k10_b0.5_g0.5.csv    # Single-row summary (per config)
└── plots/                               # Visualizations
    ├── step_scores_k10_b0.5_g0.5.png
    └── similarity_k10_b0.5_g0.5.png
```

### Aggregated Results

```
results/
├── all_datasets_summary.csv              # Master file: all datasets, all configs
├── verifier_selection_all_datasets.csv   # Stage 1 results only
├── weaver_evaluation_all_datasets.csv    # Stage 2 results only
├── best_configs_per_dataset.csv          # Best config per dataset
├── verifier_frequency_analysis.csv       # Verifier usage statistics
├── summary_statistics.json               # Overall stats
└── execution_log.txt                     # Detailed execution log
```

### all_datasets_summary.csv Columns

Key columns in the master summary file:

- `dataset_name` - Dataset identifier
- `dataset_type` - MATH-500, GPQA, or MMLU
- `model_size` - 8B or 70B
- `k`, `alpha`, `beta`, `gamma` - Hyperparameters
- `test_select_accuracy` - **Main metric** (accuracy of Weaver)
- `train_select_accuracy` - Training accuracy
- `num_verifiers` - Number of verifiers used
- `verifiers` - Comma-separated verifier names
- `greedy_score` - Stage 1 greedy selection score
- `train_file`, `test_file` - Paths to detailed results

## Advanced Usage

### Run Specific Stages Only

```bash
# Stage 1 only (verifier selection)
python verifier_selection/run_verifier_selection.py \
    --dataset_path wfang11/math500-llama70b-1-5-94-dev \
    --output_dir results/math500-llama70b \
    --k 5 10 15 \
    --beta 0.25 0.5 1.0 \
    --gamma 0.25 0.5 1.0

# Stage 2 only (Weaver evaluation)
python selection/run_weaver_evaluation.py \
    --dataset_path wfang11/math500-llama70b-1-5-94-dev \
    --output_dir results/math500-llama70b \
    --verifier_subset GRM_scores QwenPRM_avg_scores INFORM_scores \
    --k 10 --beta 0.5 --gamma 0.5
```

### Custom Hyperparameter Search

Edit `datasets_config.yaml`:

```yaml
hyperparameters:
  k: [5, 10]                  # Reduce k values
  alpha: [1.0]                # Keep alpha fixed
  beta: [0.5, 1.0]            # Reduce beta values
  gamma: [0.5, 1.0]           # Reduce gamma values
```

This reduces from 27 to 8 configurations per dataset.

### Disable WandB

Edit `datasets_config.yaml`:

```yaml
wandb:
  enabled: false
```

Or set environment variable:
```bash
export WANDB_MODE=disabled
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. **Reduce batch size** in Weaver config
2. **Run fewer datasets at once** using `--datasets` flag
3. **Use smaller model** (8B instead of 70B)

### Dataset Not Found

Verify dataset exists on HuggingFace:
```bash
python -c "from datasets import load_dataset; ds = load_dataset('wfang11/math500-llama70b-1-5-94-dev'); print(len(ds['data']))"
```

### Resume After Failure

The pipeline automatically resumes. Check log:
```bash
cat results/execution_log.txt | grep COMPLETED
```

To force restart:
```bash
python run_all_datasets_grid_search.py --config datasets_config.yaml --no-resume
```

## Expected Results

Based on the comprehensive grid search, you should see:

**Test accuracies by dataset type:**
- MATH-500 (70B): ~0.90-0.95
- MATH-500 (8B): ~0.75-0.85
- GPQA (70B): ~0.45-0.55
- GPQA (8B): ~0.35-0.45
- MMLU (70B): ~0.85-0.90
- MMLU (8B): ~0.70-0.80

**Most frequently selected verifiers:**
- QwenPRM variants (min/max/avg)
- EurusPRM Stage 1/2
- DeepSeek judges
- INFORM (70B datasets)
- GRM, GPM, URM

**Best hyperparameters (typical):**
- k: 10-15 verifiers
- beta: 0.5-1.0 (moderate to high similarity penalty)
- gamma: 0.5-1.0 (moderate to high cost penalty)

## Next Steps

After completing the grid search:

1. **Review report** - `results/report.md`
2. **Analyze best configs** - `results/best_configs_per_dataset.csv`
3. **Check WandB** - https://wandb.ai/329a/verification
4. **Compare with baselines** - Use existing `selection/run.py` with majority vote
5. **Paper results** - Export tables and figures from report

## Questions?

Common questions:

**Q: How long does it take?**
A: ~30-60 min per dataset, so 9-18 hours for all 18 sequentially.

**Q: Can I run in parallel?**
A: The current script is sequential. For parallel, run multiple instances with different `--datasets` flags.

**Q: What if I only care about 70B models?**
A: Edit `datasets_config.yaml` and remove the 8B datasets.

**Q: Can I add more datasets?**
A: Yes! Add them to `datasets_config.yaml` following the same format.

**Q: How much disk space needed?**
A: ~5-10 GB for all results (depends on dataset sizes and number of configs).

## Citation

If you use this grid search pipeline, please cite the Weaver paper:

```bibtex
@article{weaver2024,
  title={Weaver: Foundation Models for Reasoning Verification},
  author={...},
  year={2024}
}
```


#!/usr/bin/env python3
"""
Run Weaver evaluation using the paper's filtered verifier subsets.
This creates the "Subsetted Weaver" baseline for comparison.
"""
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from selection.run_weaver_evaluation import run_weaver_evaluation


def load_config(config_path: str = "datasets_config.yaml") -> dict:
    """Load the datasets configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_subset_baseline(
    datasets_to_run: list = None,
    no_wandb: bool = False,
    config_path: str = "datasets_config.yaml"
):
    """
    Run Weaver with paper's filtered subsets for each dataset.
    
    Args:
        datasets_to_run: List of dataset names to process (None = all)
        no_wandb: Disable WandB logging
        config_path: Path to datasets_config.yaml
    """
    config = load_config(config_path)
    datasets = config['datasets']
    verifier_subsets = config.get('verifier_subsets', {})
    
    # Filter datasets if specified
    if datasets_to_run:
        datasets = [d for d in datasets if d['name'] in datasets_to_run]
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    print("="*80)
    print("RUNNING SUBSETTED WEAVER BASELINE (Paper's Filtered Verifiers)")
    print("="*80)
    print(f"Datasets: {len(datasets)}")
    print(f"WandB: {'disabled' if no_wandb else 'enabled'}")
    print("="*80)
    
    for dataset_config in tqdm(datasets, desc="Processing datasets"):
        name = dataset_config['name']
        dev_path = dataset_config['dev']
        model_size = dataset_config['model_size']
        
        print(f"\n{'='*80}")
        print(f"Dataset: {name} ({model_size})")
        print(f"Path: {dev_path}")
        
        # Get paper's verifier subset for this dataset
        subset = verifier_subsets.get(name, [])
        
        if not subset:
            print(f"⚠️  No verifier subset defined for {name}, skipping")
            continue
        
        print(f"Verifier subset: {len(subset)} verifiers")
        print(f"  {subset[:3]}... (showing first 3)")
        print(f"{'='*80}")
        
        # Create output directory
        output_dir = results_dir / name
        output_dir.mkdir(exist_ok=True)
        
        # Check if already completed
        baseline_file = output_dir / "subset_weaver_baseline.csv"
        if baseline_file.exists():
            print(f"✅ Already completed: {baseline_file}")
            existing = pd.read_csv(baseline_file)
            all_results.append(existing.iloc[0].to_dict())
            continue
        
        # WandB config
        wandb_config = {
            "entity": "329a",
            "project": "verification",
            "name": f"subset-weaver-{name}"
        } if not no_wandb else None
        
        try:
            # Run Weaver with paper's subset
            df_train, df_test, summary_path = run_weaver_evaluation(
                dataset_path=dev_path,
                output_dir=str(output_dir),
                verifier_subset=subset,
                k=len(subset),  # k = number of verifiers in subset
                alpha=None,     # Not applicable for subset baseline
                beta=None,
                gamma=None,
                wandb_enabled=not no_wandb,
                wandb_config=wandb_config
            )
            
            # Read summary and add baseline info
            summary_df = pd.read_csv(summary_path)
            
            # Create baseline summary with approach label
            baseline_summary = {
                "dataset_name": name,
                "approach": "Subsetted Weaver",
                "num_verifiers": len(subset),
                "train_select_accuracy": summary_df['train_select_accuracy'].iloc[0],
                "test_select_accuracy": summary_df['test_select_accuracy'].iloc[0],
                "train_sample_accuracy": summary_df['train_sample_accuracy'].iloc[0],
                "test_sample_accuracy": summary_df['test_sample_accuracy'].iloc[0],
                "verifiers": ", ".join(subset),
                "model_size": model_size
            }
            
            # Save baseline-specific file
            pd.DataFrame([baseline_summary]).to_csv(baseline_file, index=False)
            all_results.append(baseline_summary)
            
            print(f"✅ Saved: {baseline_file}")
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_file = results_dir / "subset_weaver_all_datasets.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n{'='*80}")
        print(f"✅ Combined results saved to: {combined_file}")
        print(f"{'='*80}")
        
        # Print summary table
        print("\n" + "="*100)
        print("SUBSETTED WEAVER BASELINE SUMMARY")
        print("="*100)
        print(f"{'Dataset':<20} {'#V':<6} {'Train_Sel':<12} {'Test_Sel':<12} {'Train_Samp':<12} {'Test_Samp':<12}")
        print("-"*100)
        for r in all_results:
            print(f"{r['dataset_name']:<20} {r['num_verifiers']:<6} {r['train_select_accuracy']:<12.3f} {r['test_select_accuracy']:<12.3f} {r['train_sample_accuracy']:<12.3f} {r['test_sample_accuracy']:<12.3f}")
        print("="*100)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Weaver with paper's filtered verifier subsets"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        type=str, 
        default=None,
        help="Specific datasets to run (default: all)"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="datasets_config.yaml",
        help="Path to datasets config file"
    )
    
    args = parser.parse_args()
    
    run_subset_baseline(
        datasets_to_run=args.datasets,
        no_wandb=args.no_wandb,
        config_path=args.config
    )


if __name__ == "__main__":
    main()


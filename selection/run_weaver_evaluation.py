"""
Reusable Weaver evaluation module for grid search across datasets.
Wrapper around selection/run.py for programmatic execution.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from weaver.dataset import VerificationDataset
from weaver.models import Model
from selection.run import train_and_evaluate


def run_weaver_evaluation(
    dataset_path: str,
    output_dir: str,
    verifier_subset: Optional[List[str]] = None,
    k: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    wandb_enabled: bool = True,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run Weaver evaluation on a dataset with optional verifier subset.
    
    Args:
        dataset_path: HuggingFace dataset path
        output_dir: Directory to save results
        verifier_subset: List of verifier names to use (None = use all)
        k: Number of verifiers (for logging)
        alpha: Utility weight (for logging)
        beta: Similarity penalty weight (for logging)
        gamma: Cost penalty weight (for logging)
        wandb_enabled: Whether to log to WandB
        wandb_config: WandB configuration dict
        **kwargs: Additional configuration overrides
    
    Returns:
        df_train: Training set results
        df_test: Test set results
        output_path: Path to saved results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"WEAVER EVALUATION: {dataset_path}")
    if verifier_subset:
        print(f"Using {len(verifier_subset)} selected verifiers")
    print(f"{'='*80}\n")
    
    # Create configuration
    # VerificationDataset needs either (dataset_name, model_size) OR (dataset_path)
    # If dataset_path is provided, we'll use it directly
    # IMPORTANT: Use "specific_subset" when verifier_subset is provided, otherwise "all"
    config = {
        "verifier_cfg": {
            "verifier_type": "specific_subset" if verifier_subset else "all",
            "verifier_size": "all",
            "verifier_subset": verifier_subset if verifier_subset else []
        },
        "data_cfg": {
            "dataset_name": None,  # Will be inferred from dataset_path
            "model_size": None,    # Will be inferred from dataset_path
            "dataset_path": dataset_path,
            "train_split": 1.0,
            "train_queries": 1,
            "train_samples": 1,
            "same_train_test": False,
            "nan_replacement": 0,
            "random_seed": 0,
            "reward_threshold": 0.95,  # Binarization threshold for Weaver (matching MATH-500_70B.yaml)
            "save_weaver_scores": False,
            "normalize_type": "all_problems",
            "normalize_method": "minmax",
            "normalize_params": {},
            "closest_train_problem_method": "mean_verifier_distance",
            "closest_train_problem_metric_type": "euclidean",
            "verifier_cfg": {
                "verifier_type": "specific_subset" if verifier_subset else "all",
                "verifier_size": "all",
                "verifier_subset": verifier_subset if verifier_subset else []
            },
            "mv_as_verifier": False,
            "fixed_test_split": None
        },
        "model_cfg": {
            "model_type": "weak_supervision",
            "model_class": "per_dataset",
            "model_params": {
                "k": 2,
                "seed": 0,
                "binarize_threshold": 0.5,
                "metric": "scores",
                "n_epochs": 1000,
                "mu_epochs": 1000,
                "log_train_every": 1000,
                "lr": 0.00001,
                "use_deps": "drop",
                "use_label_on_test": True,
                "drop_imbalanced_verifiers": "all",
                "drop_k": 100,
                "cb_args": {
                    "class_balance": "labels"
                }
            }
        },
        "fit_cfg": {
            "fit_type": "wclosest_to_train"
        },
        "logging": "wandb" if wandb_enabled else "none",
        "k": int(k) if k is not None else None,
        "alpha": float(alpha) if alpha is not None else None,
        "beta": float(beta) if beta is not None else None,
        "gamma": float(gamma) if gamma is not None else None,
        "debug": False
    }
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if "." in key:
            # Handle nested keys like "data_cfg.train_split"
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    # Convert to OmegaConf
    args = OmegaConf.create(config)
    
    # Initialize WandB if enabled
    if wandb_enabled:
        if wandb_config is None:
            wandb_config = {
                "entity": "329a",
                "project": "verification",
                "name": f"weaver_{Path(dataset_path).stem}"
            }
            if k is not None:
                wandb_config["name"] += f"_k{k}"
            if beta is not None and gamma is not None:
                wandb_config["name"] += f"_b{beta}_g{gamma}"
        
        wandb_cfg = {k: v for k, v in wandb_config.items() 
                     if v is not None and v != "null" and v != ""}
        wandb.init(**wandb_cfg, config=OmegaConf.to_container(args, resolve=True))
    
    # Load data
    print("Loading dataset...")
    
    # Infer dataset_name and model_size from the dataset_path for proper handling
    # e.g., "wfang11/math500-llama8b-1-5-94-dev" -> dataset_name="MATH-500", model_size="8B"
    dataset_path_lower = dataset_path.lower()
    
    if "math500" in dataset_path_lower or "math-500" in dataset_path_lower:
        inferred_dataset_name = "MATH-500"
    elif "gpqa" in dataset_path_lower:
        inferred_dataset_name = "GPQA"
    elif "mmlu" in dataset_path_lower:
        inferred_dataset_name = "MMLU"
    else:
        inferred_dataset_name = None
        print(f"  Warning: Could not infer dataset_name from path: {dataset_path}")
    
    if "70b" in dataset_path_lower:
        inferred_model_size = "70B"
    elif "8b" in dataset_path_lower:
        inferred_model_size = "8B"
    else:
        inferred_model_size = None
        print(f"  Warning: Could not infer model_size from path: {dataset_path}")
    
    print(f"  Inferred: dataset_name={inferred_dataset_name}, model_size={inferred_model_size}")
    
    data_cfg_dict = dict(args.data_cfg)
    # Remove dataset_name and model_size from dict to avoid duplicate argument error
    data_cfg_dict.pop('dataset_name', None)
    data_cfg_dict.pop('model_size', None)
    
    data = VerificationDataset(
        dataset_name=inferred_dataset_name,
        model_size=inferred_model_size,
        **data_cfg_dict
    )
    
    print(f"  Train problems: {data.train_data[0].shape[0]}")
    print(f"  Test problems: {data.test_data[0].shape[0]}")
    print(f"  Verifiers: {len(data.verifier_names)}")
    
    # Create model
    print("\nInitializing Weaver model...")
    clusters = None  # Not using clustering for per_dataset
    num_models = None
    
    # Binarize verifiers if reward_threshold is set
    if args.data_cfg.reward_threshold is not None:
        print(f"Binarizing verifiers with threshold: {args.data_cfg.reward_threshold}")
        data.binarize_verifiers(clusters, split="train")
        data.binarize_verifiers(clusters, split="test")
    
    model = Model(data.verifier_names, clusters, **args.model_cfg, num_models=num_models)
    
    # Train and evaluate
    print("\nTraining and evaluating...")
    df_train, df_test = train_and_evaluate(data, model, args.fit_cfg)
    
    # Calculate summary metrics
    try:
        test_select_acc = df_test['top1_positive'].mean()
        train_select_acc = df_train['top1_positive'].mean()
        test_sample_acc = df_test['sample_accuracy'].mean()
        train_sample_acc = df_train['sample_accuracy'].mean()
    except:
        test_select_acc = np.nan
        train_select_acc = np.nan
        test_sample_acc = np.nan
        train_sample_acc = np.nan
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Test Select Accuracy:  {test_select_acc:.3f}")
    print(f"Train Select Accuracy: {train_select_acc:.3f}")
    print(f"Test Sample Accuracy:  {test_sample_acc:.3f}")
    print(f"Train Sample Accuracy: {train_sample_acc:.3f}")
    print(f"{'='*80}\n")
    
    # Log to WandB
    if wandb_enabled and wandb.run:
        wandb.log({
            "test_select_accuracy": test_select_acc,
            "train_select_accuracy": train_select_acc,
            "test_sample_accuracy": test_sample_acc,
            "train_sample_accuracy": train_sample_acc,
            "num_verifiers": len(data.verifier_names),
            "verifiers": data.verifier_names
        })
    
    # Save results
    config_suffix = ""
    if k is not None:
        config_suffix += f"_k{k}"
    if beta is not None:
        config_suffix += f"_b{beta}"
    if gamma is not None:
        config_suffix += f"_g{gamma}"
    
    train_file = output_dir / f"weaver_train{config_suffix}.csv"
    test_file = output_dir / f"weaver_test{config_suffix}.csv"
    
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    print(f"✅ Results saved to:")
    print(f"   Train: {train_file}")
    print(f"   Test:  {test_file}")
    
    # Create summary row
    summary = {
        "dataset_path": dataset_path,
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "test_select_accuracy": test_select_acc,
        "train_select_accuracy": train_select_acc,
        "test_sample_accuracy": test_sample_acc,
        "train_sample_accuracy": train_sample_acc,
        "num_verifiers": len(data.verifier_names),
        "verifiers": ", ".join(data.verifier_names) if data.verifier_names else "",
        "train_file": str(train_file),
        "test_file": str(test_file)
    }
    
    summary_file = output_dir / f"weaver_summary{config_suffix}.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    
    # Finish WandB
    if wandb_enabled and wandb.run:
        wandb.finish()
    
    return df_train, df_test, str(summary_file)


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Weaver evaluation")
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="HuggingFace dataset path")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--verifier_subset", nargs="+", type=str, default=None,
                      help="List of verifier names to use")
    parser.add_argument("--k", type=int, default=None,
                      help="Number of verifiers (for logging)")
    parser.add_argument("--alpha", type=float, default=None,
                      help="Utility weight (for logging)")
    parser.add_argument("--beta", type=float, default=None,
                      help="Similarity penalty weight (for logging)")
    parser.add_argument("--gamma", type=float, default=None,
                      help="Cost penalty weight (for logging)")
    parser.add_argument("--no-wandb", action="store_true",
                      help="Disable WandB logging")
    parser.add_argument("--wandb-entity", type=str, default="329a",
                      help="WandB entity")
    parser.add_argument("--wandb-project", type=str, default="verification",
                      help="WandB project")
    
    args = parser.parse_args()
    
    wandb_config = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": f"weaver_{Path(args.dataset_path).stem}"
    } if not args.no_wandb else None
    
    run_weaver_evaluation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        verifier_subset=args.verifier_subset,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        wandb_enabled=not args.no_wandb,
        wandb_config=wandb_config
    )


if __name__ == "__main__":
    main()


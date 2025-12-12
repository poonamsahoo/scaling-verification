"""
Reusable verifier selection module for grid search across datasets.
Refactored from search.py to support programmatic execution.
"""
import sys
from pathlib import Path
# Add verifier_selection directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import utils
import datasets
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


def plot_selected_similarity(selected_indices, similarities, save_path=None):
    """Plot similarity heatmap for selected verifiers."""
    sim_sub = similarities['pearson'][np.ix_(selected_indices, selected_indices)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(sim_sub, vmin=-1, vmax=1, cmap="vlag", square=True, cbar=True, ax=ax)
    ax.set_title("Selected Pearson Similarity")
    ax.set_xlabel("Verifiers")
    ax.set_ylabel("Verifiers")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_step_scores(step_scores, save_path=None):
    """Plot greedy step scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(step_scores) + 1), step_scores, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Greedy score")
    ax.set_title("Greedy step scores")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def run_verifier_selection(
    dataset_path: str,
    output_dir: str,
    k_values: List[int] = [5, 10, 15],
    alpha: float = 1.0,
    beta_values: List[float] = [0.25, 0.5, 1.0],
    gamma_values: List[float] = [0.25, 0.5, 1.0],
    wandb_enabled: bool = True,
    wandb_config: Optional[Dict] = None,
    save_plots: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Run verifier selection grid search on a dataset.
    
    Args:
        dataset_path: HuggingFace dataset path (e.g., "wfang11/math500-llama70b-1-5-94-dev")
        output_dir: Directory to save results
        k_values: List of k values (number of verifiers to select)
        alpha: Utility weight (typically fixed at 1.0)
        beta_values: List of similarity penalty weights
        gamma_values: List of cost penalty weights
        wandb_enabled: Whether to log to WandB
        wandb_config: WandB configuration dict
        save_plots: Whether to save plots
    
    Returns:
        results_df: DataFrame with all configurations and results
        output_csv_path: Path to saved results CSV
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dev_ds = datasets.load_dataset(dataset_path)["data"]
    
    # Extract scores and compute metrics
    print("Extracting verifier scores...")
    scores_matrix, verifier_names = utils.extract_scores_matrix(dev_ds)
    
    print("Computing similarities...")
    similarities = utils.similarities_dict(scores_matrix)
    
    print("Computing utilities...")
    utilities = utils.utilities_dict(scores_matrix, dev_ds, verifier_names)
    
    print("Computing parameter counts...")
    param_counts = utils.costs_dict(verifier_names)
    
    # Initialize WandB if enabled
    if wandb_enabled:
        if wandb_config is None:
            wandb_config = {
                "entity": "329a",
                "project": "verification",
                "name": f"verifier_selection_{Path(dataset_path).stem}"
            }
        wandb.init(**wandb_config)
    
    # Run grid search
    print("\n" + "="*80)
    print("STARTING HYPERPARAMETER SEARCH")
    print("="*80)
    
    results = []
    
    for k in k_values:
        for beta in beta_values:
            for gamma in gamma_values:
                print(f"\nRunning: k={k}, alpha={alpha}, beta={beta}, gamma={gamma}")
                
                # Run greedy selection
                sel = utils.greedy_select(
                    utilities,
                    similarities['pearson'],
                    verifier_names,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    param_counts=param_counts,
                )
                
                selected_indices = sel["order"]
                selected_names = sel["verifiers"]
                step_scores = sel["step_scores"]
                total_score = float(np.sum(step_scores))
                param_counts_gb = list(sel["param_counts"])
                
                # Log to WandB
                if wandb_enabled:
                    wandb.log({
                        "k": k,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "total_greedy_score": total_score,
                        "final_step_score": step_scores[-1],
                        "selected_param_total_GB": float(np.sum(param_counts_gb)),
                    })
                    
                    # Log plots
                    if save_plots:
                        fig1 = plot_step_scores(step_scores)
                        fig2 = plot_selected_similarity(selected_indices, similarities)
                        wandb.log({
                            "step_scores": wandb.Image(fig1),
                            "selected_similarity": wandb.Image(fig2),
                        })
                        plt.close(fig1)
                        plt.close(fig2)
                
                # Save plots locally if requested
                if save_plots and not wandb_enabled:
                    plots_dir = output_dir / "plots"
                    plots_dir.mkdir(exist_ok=True)
                    
                    fig1 = plot_step_scores(
                        step_scores,
                        save_path=plots_dir / f"step_scores_k{k}_b{beta}_g{gamma}.png"
                    )
                    fig2 = plot_selected_similarity(
                        selected_indices,
                        similarities,
                        save_path=plots_dir / f"similarity_k{k}_b{beta}_g{gamma}.png"
                    )
                    plt.close(fig1)
                    plt.close(fig2)
                
                # Accumulate results
                results.append({
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "total_greedy_score": total_score,
                    "final_step_score": step_scores[-1],
                    "selected_verifiers": ", ".join(selected_names),
                    "selected_param_counts_GB": ", ".join(f"{x:.1f}" for x in param_counts_gb),
                    "num_verifiers": len(selected_names),
                    "total_param_GB": float(np.sum(param_counts_gb))
                })
                
                print(f"  Total score: {total_score:.3f}")
                print(f"  Selected: {', '.join(selected_names[:3])}..." if len(selected_names) > 3 else f"  Selected: {', '.join(selected_names)}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_csv = output_dir / "verifier_selection_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    # Log summary table to WandB
    if wandb_enabled:
        wandb.log({"hparam_results": wandb.Table(dataframe=df)})
        wandb.finish()
    
    print("\n" + "="*80)
    print("VERIFIER SELECTION COMPLETE")
    print("="*80)
    
    return df, str(output_csv)


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run verifier selection grid search")
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="HuggingFace dataset path (e.g., wfang11/math500-llama70b-1-5-94-dev)")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 15],
                      help="K values (number of verifiers)")
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Utility weight")
    parser.add_argument("--beta", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                      help="Similarity penalty weights")
    parser.add_argument("--gamma", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                      help="Cost penalty weights")
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
        "name": f"verifier_selection_{Path(args.dataset_path).stem}"
    } if not args.no_wandb else None
    
    run_verifier_selection(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        k_values=args.k,
        alpha=args.alpha,
        beta_values=args.beta,
        gamma_values=args.gamma,
        wandb_enabled=not args.no_wandb,
        wandb_config=wandb_config
    )


if __name__ == "__main__":
    main()


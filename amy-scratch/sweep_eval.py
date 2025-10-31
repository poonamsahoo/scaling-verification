import argparse
import os
import subprocess
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from verifier_selection import utils


def plot_selected_similarity(similarities: dict, selected_indices: list[int]):
    sim_sub = similarities['pearson'][np.ix_(selected_indices, selected_indices)]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(sim_sub, vmin=-1, vmax=1, cmap="vlag", square=True, cbar=True, ax=ax)
    ax.set_title("Selected Pearson Similarity")
    ax.set_xlabel("Verifiers")
    ax.set_ylabel("Verifiers")
    plt.tight_layout()
    return fig


def plot_step_scores(step_scores: list[float]):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(step_scores) + 1), step_scores, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Greedy score")
    ax.set_title("Greedy step scores")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_name", type=str, required=True, help="Base HF dataset repo (e.g., amyguan/math500-k50-80-10-10)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--config_name", type=str, default="new_test", help="selection.run Hydra config name")
    parser.add_argument("--log_dataset_name", type=str, required=True)
    parser.add_argument("--log_dataset_dev", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="verification")
    parser.add_argument("--wandb_entity", type=str, default="329a")
    args = parser.parse_args()

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     config={
                         "hub_name": args.hub_name,
                         "alpha": args.alpha,
                         "beta": args.beta,
                         "gamma": args.gamma,
                         "k": args.k,
                         "config_name": args.config_name,
                         "log_dataset_name": args.log_dataset_name,
                         "log_dataset_dev": args.log_dataset_dev,
                     },
                     job_type="verifier_sweep")

    # Load splits (dev used for selection)
    dev_ds = datasets.load_dataset(f"{args.hub_name}-dev")["data"]

    # Compute metrics
    scores_matrix, verifier_names = utils.extract_scores_matrix(dev_ds)
    similarities = utils.similarities_dict(scores_matrix)
    utilities = utils.utilities_dict(scores_matrix, dev_ds, verifier_names)
    param_counts = utils.costs_dict(verifier_names)

    # Greedy selection
    sel = utils.greedy_select(
        utilities,
        similarities['pearson'],
        verifier_names,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        param_counts=param_counts,
    )

    selected_indices = sel["order"]
    selected_names = sel["verifiers"]
    step_scores = sel["step_scores"]
    total_score = float(np.sum(step_scores))
    param_counts_gb = list(sel["param_counts"])  # already in billions

    # Log selection artifacts
    wandb.log({
        "total_greedy_score": total_score,
        "final_step_score": step_scores[-1],
        "param_counts_gb": param_counts_gb,
        "selected_param_total_GB": float(np.sum(param_counts_gb)),
        "selected_verifiers": selected_names,
    })

    fig1 = plot_step_scores(step_scores)
    fig2 = plot_selected_similarity(similarities, selected_indices)
    wandb.log({
        "step_scores": wandb.Image(fig1),
        "selected_similarity": wandb.Image(fig2),
    })
    plt.close(fig1)
    plt.close(fig2)

    # Optionally, propagate group/name to evaluation run
    # group = run.group or f"verifier_sweep"
    base_name = f"{args.log_dataset_name}-{args.log_dataset_dev}-k{args.k}-a{args.alpha}-b{args.beta}-g{args.gamma}"

    # Kick off evaluation run (Hydra overrides)
    hydra_list = "[" + ",".join(f"'{x}'" for x in selected_names) + "]"
    cmd = [
        "python", "selection/run.py",
        "--config-name", args.config_name,
        f"data_cfg.dataset_path={args.hub_name}-val",
        f"log_dataset_name={args.log_dataset_name}",
        f"log_dataset_dev={args.log_dataset_dev}",
        f"verifier_cfg.verifier_subset={hydra_list}",
        # f"wandb_cfg.group={group}",
        f"wandb_cfg.name={base_name}",
        f"wandb_cfg.id={run.id}",
        f"wandb_cfg.resume=allow",
    ]
    # Ensure selection/run.py logs to same project/entity
    if args.wandb_entity:
        cmd.append(f"wandb_cfg.entity={args.wandb_entity}")
    cmd.append(f"wandb_cfg.project={args.wandb_project}")

    # Inherit env; silent logging is already handled in selection.run
    subprocess.run(cmd, check=True)

    wandb.finish()


if __name__ == "__main__":
    main()



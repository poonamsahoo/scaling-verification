import utils
import datasets
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    hub_name = "amyguan/math500-k50-80-10-10"
    dev_ds = datasets.load_dataset(f"{hub_name}-dev")["data"]
    val_ds = datasets.load_dataset(f"{hub_name}-val")["data"]
    # test_ds = datasets.load_dataset(f"{hub_name}-test")["data"]

    ### COMPUTE DEV METRICS ###
    scores_matrix, verifier_names = utils.extract_scores_matrix(dev_ds)
    similarities = utils.similarities_dict(scores_matrix)
    utilities = utils.utilities_dict(scores_matrix, dev_ds, verifier_names)
    param_counts = utils.costs_dict(verifier_names)

    ### INIT WANDB ###
    wandb.init(entity="329a", project="verification", name="verifier_hparam_search")

    ### HYPERPARAMETER SEARCH ###
    alpha = 1.0
    results = []

    def plot_selected_similarity(selected_indices):
        sim_sub = similarities['pearson'][np.ix_(selected_indices, selected_indices)]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(sim_sub, vmin=-1, vmax=1, cmap="vlag", square=True, cbar=True, ax=ax)
        ax.set_title("Selected Pearson Similarity")
        ax.set_xlabel("Verifiers")
        ax.set_ylabel("Verifiers")
        plt.tight_layout()
        return fig

    def plot_step_scores(step_scores):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(step_scores) + 1), step_scores, marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Greedy score")
        ax.set_title("Greedy step scores")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    for k in [5, 10, 15]:
        for beta in [0.25, 0.5, 1.0]:
            for gamma in [0.25, 0.5, 1.0]:
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
                param_counts_gb = list(sel["param_counts"])  # already in billions

                # Log numbers
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
                fig1 = plot_step_scores(step_scores)
                fig2 = plot_selected_similarity(selected_indices)
                wandb.log({
                    "step_scores": wandb.Image(fig1),
                    "selected_similarity": wandb.Image(fig2),
                })
                plt.close(fig1)
                plt.close(fig2)

                # Accumulate table rows
                results.append({
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "total_greedy_score": total_score,
                    "final_step_score": step_scores[-1],
                    "selected_verifiers": ", ".join(selected_names),
                    "selected_param_counts_GB": ", ".join(f"{x:.1f}" for x in param_counts_gb),
                })

    # Log a summary table
    df = pd.DataFrame(results)
    df.to_csv("verifier_hparam_search_results.csv", index=False)
    wandb.log({"hparam_results": wandb.Table(dataframe=df)})

    wandb.finish()

    

    
if __name__ == "__main__":
    main()
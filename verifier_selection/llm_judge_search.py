import utils
import datasets
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# have cluster_id has a column in the hub_name dataset cluster_id
# want to run a hyperparameter search for each cluster. that means loading in the dataset, and then filtering to the given cluster_id, and then doing the dev metrics/verifier selection process. and i want all the results to be saved to a single csv file. 

def main():
    hub_name = "pnsahoo/GPQA-llama8b-qwen-judge"
    dev_ds = datasets.load_dataset(f"{hub_name}-dev")["data"]
    val_ds = datasets.load_dataset(f"{hub_name}-val")["data"]
    test_ds = datasets.load_dataset(f"{hub_name}-test")["data"]

    # Check if cluster_id column exists
    if "cluster_id" not in dev_ds[0].keys():
        raise ValueError("cluster_id column not found in dataset. Please ensure the dataset has a cluster_id column.")
    
    # Get all unique cluster IDs
    unique_cluster_ids = sorted(set(dev_ds["cluster_id"]))
    print(f"Found {len(unique_cluster_ids)} unique clusters: {unique_cluster_ids}")

    ### INIT WANDB ###
    wandb.init(entity="329a", project="llm-judge-augmented-weaver", name="GPQA-llama8b-qwen-judge-amended-verifier_hparam_search_10percent")

    ### HYPERPARAMETER SEARCH ###
    alpha = 1.0
    results = []

    def plot_selected_similarity(selected_indices, similarities):
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

    # Loop over each cluster
    for cluster_id in unique_cluster_ids:
        print(f"\n{'='*60}")
        print(f"Processing cluster_id: {cluster_id}")
        print(f"{'='*60}")
        
        # Filter dataset to current cluster
        cluster_dev_ds = dev_ds.filter(lambda x: x["cluster_id"] == cluster_id)
        print(f"Filtered dataset size: {len(cluster_dev_ds)} samples")
        
        if len(cluster_dev_ds) == 0:
            print(f"Warning: No samples found for cluster_id {cluster_id}, skipping...")
            continue

        ### COMPUTE DEV METRICS FOR THIS CLUSTER ###
        scores_matrix, verifier_names = utils.extract_scores_matrix(cluster_dev_ds)
        similarities = utils.similarities_dict(scores_matrix)
        utilities = utils.utilities_dict(scores_matrix, cluster_dev_ds, verifier_names)
        param_counts = utils.costs_dict(verifier_names)

        for k in [5, 10, 15]:
            for beta in [0.25, 0.5, 1.0]:
                for gamma in [0.25, 0.5, 1.0]:
        # for k in [10]:
        #     for beta in [1.0]:
        #         for gamma in [0.5, 1.0]:
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
                        "cluster_id": cluster_id,
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
                    fig2 = plot_selected_similarity(selected_indices, similarities)
                    wandb.log({
                        f"cluster_{cluster_id}_step_scores": wandb.Image(fig1),
                        f"cluster_{cluster_id}_selected_similarity": wandb.Image(fig2),
                    })
                    plt.close(fig1)
                    plt.close(fig2)

                    # Accumulate table rows with cluster_id
                    results.append({
                        "cluster_id": cluster_id,
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
    df.to_csv("results/GPQA-llama8b-qwen-judge-amended-verifier_hparam_search_results_10percent.csv", index=False)
    wandb.log({"hparam_results": wandb.Table(dataframe=df)})

    wandb.finish()

    

    
if __name__ == "__main__":
    main()
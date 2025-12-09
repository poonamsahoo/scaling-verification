import datasets
import utils
import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def extract_features_from_dataset(dataset, verifier_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features (mean and variance of verifier scores) and difficulty labels from dataset.
    
    Args:
        dataset: HuggingFace dataset with verifier scores and answer_correct
        verifier_names: List of verifier column names
    
    Returns:
        X_features: (num_problems, 2 * num_verifiers) - mean and var of verifier scores
        y_difficulty: (num_problems,) - mean correctness (difficulty) per problem
    """
    # Extract verifier scores for each problem
    # Shape: (num_problems, num_samples, num_verifiers)
    X_data = []
    y_data = []
    
    for problem in dataset:
        # Get verifier scores for this problem
        problem_scores = []
        for verifier_name in verifier_names:
            scores = problem[verifier_name]  # List of scores for each sample
            problem_scores.append(scores)
        
        # Stack: (num_samples, num_verifiers)
        problem_scores = np.array(problem_scores).T
        X_data.append(problem_scores)
        
        # Get correctness labels
        y_data.append(problem['answer_correct'])
    
    # Convert to numpy arrays
    X_data = np.array([np.array(x) for x in X_data], dtype=object)  # Variable number of samples
    y_data = np.array([np.array(y) for y in y_data], dtype=object)
    
    # Extract features: mean and variance across samples for each problem
    mean_features = np.array([np.mean(x, axis=0) for x in X_data])  # (num_problems, num_verifiers)
    var_features = np.array([np.var(x, axis=0) for x in X_data])     # (num_problems, num_verifiers)
    X_features = np.concatenate([mean_features, var_features], axis=1)  # (num_problems, 2 * num_verifiers)
    
    # Compute difficulty (mean correctness) for each problem
    problem_difficulty = np.array([np.mean(y) for y in y_data])  # (num_problems,)
    
    return X_features, problem_difficulty


def bin_difficulty(difficulty: np.ndarray, n_clusters: int, method: str = "quantile") -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin difficulty values into n_clusters.
    
    Args:
        difficulty: (num_problems,) - mean correctness values (soft labels)
        n_clusters: Number of clusters
        method: "uniform" (equal-width bins) or "quantile" (equal-size bins)
    
    Returns:
        difficulty_labels: (num_problems,) - cluster labels (0 to n_clusters-1) - HARD LABELS
        bin_edges: Bin edges used for binning
    """
    num_problems = len(difficulty)
    
    if method == "quantile":
        # Use quantile-based binning for balanced clusters
        # For equal-sized bins, sort and assign directly to ensure exactly balanced
        sorted_indices = np.argsort(difficulty)
        difficulty_labels = np.zeros(num_problems, dtype=int)
        
        # Assign each example to a cluster based on sorted position
        # This ensures each cluster gets approximately equal number of examples
        # Formula: divide sorted indices into n_clusters groups
        examples_per_cluster = num_problems / n_clusters
        for i, idx in enumerate(sorted_indices):
            cluster_id = int(i / examples_per_cluster)
            difficulty_labels[idx] = min(cluster_id, n_clusters - 1)  # Safety check
        
        # Compute bin edges from percentiles for reference
        percentiles = np.linspace(0, 100, n_clusters + 1)
        bin_edges = np.percentile(difficulty, percentiles)
    else:  # uniform
        # Equal-width bins
        bin_edges = np.linspace(difficulty.min(), difficulty.max(), n_clusters + 1)
        difficulty_labels = np.digitize(difficulty, bins=bin_edges[:-1], right=False)
        difficulty_labels = np.clip(difficulty_labels, 0, n_clusters - 1)
    
    return difficulty_labels, bin_edges


def bin_difficulty_with_edges(difficulty: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Bin difficulty values using pre-computed bin edges (for val/test sets).
    
    Args:
        difficulty: (num_problems,) - mean correctness values
        bin_edges: Pre-computed bin edges from training set
    
    Returns:
        difficulty_labels: (num_problems,) - cluster labels (0 to n_clusters-1)
    """
    n_clusters = len(bin_edges) - 1
    difficulty_labels = np.digitize(difficulty, bins=bin_edges[:-1], right=False)
    difficulty_labels = np.clip(difficulty_labels, 0, n_clusters - 1)
    return difficulty_labels


def visualize_clustering_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    difficulty: np.ndarray,
    set_name: str,
    n_clusters: int,
    bin_edges: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Create visualizations for clustering results.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        difficulty: Difficulty values (not used, kept for compatibility)
        set_name: Name of the dataset (e.g., "dev", "val", "test")
        n_clusters: Number of clusters
        bin_edges: Bin edges used for clustering (not used, kept for compatibility)
        save_path: Optional path to save figures
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=range(n_clusters), yticklabels=range(n_clusters))
    ax1.set_xlabel('Predicted Cluster')
    ax1.set_ylabel('True Cluster')
    ax1.set_title(f'{set_name.capitalize()} Set: Confusion Matrix')
    
    # 2. Distribution Comparison
    x_pos = np.arange(n_clusters)
    width = 0.35
    true_counts = np.bincount(y_true, minlength=n_clusters)
    pred_counts = np.bincount(y_pred, minlength=n_clusters)
    ax2.bar(x_pos - width/2, true_counts, width, label='True', alpha=0.8)
    ax2.bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{set_name.capitalize()} Set: Cluster Distribution')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Clustering Results: {set_name.capitalize()} Set', fontsize=14, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def semi_supervised_naive_bayes_em(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    n_clusters: int,
    max_iter: int = 10,
    tol: float = 1e-4,
    verbose: bool = True,
    labeled_weight: float = 10.0,
    confidence_threshold: float = 0.8,
    max_pseudo_per_iter: Optional[int] = None
) -> Tuple[GaussianNB, np.ndarray, np.ndarray]:
    """
    Semi-supervised Naive Bayes with EM for difficulty clustering.
    
    Args:
        X_labeled: (num_labeled, num_features) - features from labeled data
        y_labeled: (num_labeled,) - difficulty labels (clusters) from labeled data
        X_unlabeled: (num_unlabeled, num_features) - features from unlabeled data
        n_clusters: Number of clusters
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        verbose: Print progress
        labeled_weight: Weight for labeled data (higher = more respect for labeled data)
                       Effectively duplicates labeled examples this many times
        confidence_threshold: Minimum confidence (0-1) to use pseudo-labeled examples
        max_pseudo_per_iter: Maximum pseudo-labeled examples to add per iteration (None = all confident)
    
    Returns:
        model: Trained GaussianNB model
        labeled_predictions: (num_labeled,) - predicted clusters for labeled data
        unlabeled_predictions: (num_unlabeled,) - predicted clusters for unlabeled data
    """
    # Initial training on labeled data only
    model = GaussianNB()
    model.fit(X_labeled, y_labeled)
    
    if verbose:
        print(f"Initial model trained on {len(X_labeled)} labeled examples")
    
    # EM algorithm
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iter):
        # E-step: Predict clusters for unlabeled data
        unlabeled_probs = model.predict_proba(X_unlabeled)  # (num_unlabeled, n_clusters)
        unlabeled_predictions = model.predict(X_unlabeled)
        
        # M-step: Retrain model using labeled data + pseudo-labeled unlabeled data
        # Use soft labels (probabilities) for unlabeled data
        # Combine labeled and unlabeled data
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # Create soft labels: use actual labels for labeled data, probabilities for unlabeled
        y_combined_soft = np.zeros((len(X_combined), n_clusters))
        y_combined_soft[:len(X_labeled)] = np.eye(n_clusters)[y_labeled]  # One-hot for labeled
        y_combined_soft[len(X_labeled):] = unlabeled_probs  # Probabilities for unlabeled
        
        # Retrain model (GaussianNB doesn't support soft labels directly, so we use hard pseudo-labels)
        # Use confidence threshold and labeled_weight to respect labeled data more
        confident_mask = np.max(unlabeled_probs, axis=1) >= confidence_threshold
        confident_predictions = unlabeled_predictions[confident_mask]
        confident_X = X_unlabeled[confident_mask]
        
        # Limit number of pseudo-labeled examples per iteration if specified
        if max_pseudo_per_iter is not None and len(confident_X) > max_pseudo_per_iter:
            # Select most confident examples
            confident_probs = unlabeled_probs[confident_mask]
            top_confident_indices = np.argsort(np.max(confident_probs, axis=1))[-max_pseudo_per_iter:]
            confident_X = confident_X[top_confident_indices]
            confident_predictions = confident_predictions[top_confident_indices]
        
        if len(confident_X) > 0:
            # Weight labeled data more by duplicating it
            # This makes the model respect labeled data more during training
            if labeled_weight > 1.0:
                # Duplicate labeled examples to give them more weight
                num_duplicates = int(labeled_weight)
                X_labeled_weighted = np.repeat(X_labeled, num_duplicates, axis=0)
                y_labeled_weighted = np.repeat(y_labeled, num_duplicates, axis=0)
            else:
                X_labeled_weighted = X_labeled
                y_labeled_weighted = y_labeled
            
            # Combine weighted labeled data with confident pseudo-labeled data
            X_train_combined = np.vstack([X_labeled_weighted, confident_X])
            y_train_combined = np.concatenate([y_labeled_weighted, confident_predictions])
            
            # Retrain model
            model = GaussianNB()
            model.fit(X_train_combined, y_train_combined)
            
            if verbose:
                print(f"Iteration {iteration + 1}: Using {len(confident_X)} confident pseudo-labeled examples "
                      f"(out of {len(X_unlabeled)} unlabeled), labeled_weight={labeled_weight:.1f}x")
        else:
            if verbose:
                print(f"Iteration {iteration + 1}: No confident predictions (threshold={confidence_threshold:.2f}), "
                      f"keeping previous model")
        
        # Compute log-likelihood for convergence check
        # Log-likelihood of labeled data (using true labels)
        labeled_log_probs = model.predict_log_proba(X_labeled)
        # Map y_labeled to model.classes_ indices
        label_to_class_idx = {label: idx for idx, label in enumerate(model.classes_)}
        labeled_indices = np.array([label_to_class_idx.get(label, 0) for label in y_labeled])
        labeled_log_likelihood = np.sum(labeled_log_probs[np.arange(len(X_labeled)), labeled_indices])
        
        # Log-likelihood of unlabeled data (expected log-likelihood)
        unlabeled_log_probs = model.predict_log_proba(X_unlabeled)
        unlabeled_log_likelihood = np.sum(np.sum(unlabeled_probs * unlabeled_log_probs, axis=1))
        
        log_likelihood = labeled_log_likelihood + unlabeled_log_likelihood
        
        if verbose:
            print(f"  Log-likelihood: {log_likelihood:.4f} (labeled: {labeled_log_likelihood:.4f}, "
                  f"unlabeled: {unlabeled_log_likelihood:.4f})")
        
        # Check convergence
        if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break
        
        prev_log_likelihood = log_likelihood
    
    # Final predictions
    # With proper weighting, model should respect labeled data, but check how well it does
    labeled_predictions = model.predict(X_labeled)
    unlabeled_predictions = model.predict(X_unlabeled)
    
    # Check how well model respects labeled data
    labeled_accuracy = np.mean(labeled_predictions == y_labeled)
    if verbose:
        print(f"\nFinal labeled data accuracy: {labeled_accuracy:.4f} "
              f"(how well model respects true labels)")
    
    return model, labeled_predictions, unlabeled_predictions


def fit(hub_name: str = "amyguan/math500-k50-1-5-94", n_clusters: int = 5, use_em: bool = True):
    name = hub_name.split("/")[-1]
    dev_ds = datasets.load_dataset(f"{hub_name}-dev")["data"]
    val_ds = datasets.load_dataset(f"{hub_name}-val")["data"]
    test_ds = datasets.load_dataset(f"{hub_name}-test")["data"]

    ### EXTRACT VERIFIER SCORES ###
    print("Extracting verifier scores...")
    scores_matrix, verifier_names = utils.extract_scores_matrix(dev_ds)
    print(f"Found {len(verifier_names)} verifiers")
    
    ### EXTRACT FEATURES ###
    print("\nExtracting features from datasets...")
    X_dev, y_dev_difficulty = extract_features_from_dataset(dev_ds, verifier_names)
    X_val, y_val_difficulty = extract_features_from_dataset(val_ds, verifier_names)
    X_test, y_test_difficulty = extract_features_from_dataset(test_ds, verifier_names)
    
    print(f"Dev set: {len(X_dev)} problems")
    print(f"Val set: {len(X_val)} problems")
    print(f"Test set: {len(X_test)} problems")
    print(f"Feature dimension: {X_dev.shape[1]}")
    
    ### SETUP CLUSTERING ###
    print(f"\nClustering into {n_clusters} difficulty clusters...")
    
    # Bin difficulty labels for labeled (dev) set
    # Use quantile binning to leverage continuous (soft) values for better clusters (more equally sized clusters)
    y_dev_labels, bin_edges = bin_difficulty(y_dev_difficulty, n_clusters, method="quantile")
    print(f"Dev difficulty range: [{y_dev_difficulty.min():.3f}, {y_dev_difficulty.max():.3f}]")
    print(f"Dev cluster distribution: {np.bincount(y_dev_labels)}")
    
    ### SEMI-SUPERVISED LEARNING ###
    if use_em:
        print("\n" + "="*60)
        print("Semi-supervised Naive Bayes with EM")
        print("="*60)
        
        # Combine val and test as unlabeled data
        X_unlabeled = np.vstack([X_val, X_test])
        
        # Train semi-supervised model
        # Parameters to control how much EM respects labeled data:
        # - labeled_weight: Higher = more weight on labeled data (default: 10.0 = labeled data counts 10x more)
        # - confidence_threshold: Higher = more conservative pseudo-labeling (default: 0.8)
        # - max_pseudo_per_iter: Limit pseudo-labeled examples per iteration (default: None = use all)
        model, dev_predictions, unlabeled_predictions = semi_supervised_naive_bayes_em(
            X_labeled=X_dev,
            y_labeled=y_dev_labels,
            X_unlabeled=X_unlabeled,
            n_clusters=n_clusters,
            max_iter=10,
            verbose=True,
            labeled_weight=10.0,  # Labeled data counts 10x more (adjust to respect labels more/less)
            confidence_threshold=0.8,  # Higher = more conservative (0.7-0.9 range)
            max_pseudo_per_iter=None  # None = use all confident, or set to limit (e.g., 50)
        )
        
        # Split predictions back to val and test
        val_predictions = unlabeled_predictions[:len(X_val)]
        test_predictions = unlabeled_predictions[len(X_val):]
    else:
        print("\n" + "="*60)
        print("Supervised Naive Bayes (no EM)")
        print("="*60)
        
        # Train supervised model on dev set only
        model = GaussianNB()
        model.fit(X_dev, y_dev_labels)
        
        # Predict on all sets
        dev_predictions = model.predict(X_dev)
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)
        
        print(f"Trained on {len(X_dev)} labeled examples")
    
    ### EVALUATION ###
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    # Compute distance metrics on dev set (how well model respects true labels)
    dev_l1_distance = np.mean(np.abs(dev_predictions - y_dev_labels))
    dev_l2_distance = np.sqrt(np.mean((dev_predictions - y_dev_labels) ** 2))
    dev_accuracy = np.mean(dev_predictions == y_dev_labels)
    print(f"Dev set cluster metrics:")
    print(f"  L1 distance (mean absolute error): {dev_l1_distance:.4f}")
    print(f"  L2 distance (RMSE): {dev_l2_distance:.4f}")
    print(f"  Accuracy (exact match): {dev_accuracy:.4f} ({dev_accuracy*100:.1f}%)")
    
    # Evaluate on val/test sets using bin_edges from dev set
    print("\n" + "="*60)
    print("Val/Test Set Evaluation")
    print("="*60)
    
    # Bin val/test difficulties using the same bin_edges from dev set
    y_val_labels = bin_difficulty_with_edges(y_val_difficulty, bin_edges)
    y_test_labels = bin_difficulty_with_edges(y_test_difficulty, bin_edges)
    
    # Compute distance metrics on val/test sets
    val_l1_distance = np.mean(np.abs(val_predictions - y_val_labels))
    val_l2_distance = np.sqrt(np.mean((val_predictions - y_val_labels) ** 2))
    val_accuracy = np.mean(val_predictions == y_val_labels)
    
    test_l1_distance = np.mean(np.abs(test_predictions - y_test_labels))
    test_l2_distance = np.sqrt(np.mean((test_predictions - y_test_labels) ** 2))
    test_accuracy = np.mean(test_predictions == y_test_labels)
    
    print(f"Val set cluster metrics:")
    print(f"  L1 distance (mean absolute error): {val_l1_distance:.4f}")
    print(f"  L2 distance (RMSE): {val_l2_distance:.4f}")
    print(f"  Accuracy (exact match): {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
    
    print(f"\nTest set cluster metrics:")
    print(f"  L1 distance (mean absolute error): {test_l1_distance:.4f}")
    print(f"  L2 distance (RMSE): {test_l2_distance:.4f}")
    print(f"  Accuracy (exact match): {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # Show distributions
    print(f"\nVal set cluster distribution:")
    print(f"  Predicted: {np.bincount(val_predictions, minlength=n_clusters)}")
    print(f"  True:      {np.bincount(y_val_labels, minlength=n_clusters)}")
    
    print(f"\nTest set cluster distribution:")
    print(f"  Predicted: {np.bincount(test_predictions, minlength=n_clusters)}")
    print(f"  True:      {np.bincount(y_test_labels, minlength=n_clusters)}")
    
    # # Show confusion details for val set
    # val_wrong = val_predictions != y_val_labels
    # if np.any(val_wrong):
    #     print(f"\nVal set: {np.sum(val_wrong)}/{len(val_predictions)} misclustered examples")
    #     # Show first few errors
    #     wrong_indices = np.where(val_wrong)[0][:5]  # Show first 5 errors
    #     for idx in wrong_indices:
    #         print(f"  Example {idx}: true={y_val_labels[idx]}, pred={val_predictions[idx]}, "
    #               f"difficulty={y_val_difficulty[idx]:.3f}")
    #     if len(wrong_indices) < np.sum(val_wrong):
    #         print(f"  ... and {np.sum(val_wrong) - len(wrong_indices)} more")
    # else:
    #     print(f"\n✓ All val predictions correct!")
    
    # # Show confusion details for test set
    # test_wrong = test_predictions != y_test_labels
    # if np.any(test_wrong):
    #     print(f"\nTest set: {np.sum(test_wrong)}/{len(test_predictions)} misclustered examples")
    #     # Show first few errors
    #     wrong_indices = np.where(test_wrong)[0][:5]  # Show first 5 errors
    #     for idx in wrong_indices:
    #         print(f"  Example {idx}: true={y_test_labels[idx]}, pred={test_predictions[idx]}, "
    #               f"difficulty={y_test_difficulty[idx]:.3f}")
    #     if len(wrong_indices) < np.sum(test_wrong):
    #         print(f"  ... and {np.sum(test_wrong) - len(wrong_indices)} more")
    # else:
    #     print(f"\n✓ All test predictions correct!")
    
    # Show difficulty statistics per cluster (using TRUE labels, not predictions)
    print("\nDifficulty statistics per cluster (on dev set, using true labels):")
    for cluster_id in range(n_clusters):
        cluster_mask = y_dev_labels == cluster_id  # Use true labels, not predictions
        if np.any(cluster_mask):
            cluster_difficulty = y_dev_difficulty[cluster_mask]
            print(f"  Cluster {cluster_id}: mean={cluster_difficulty.mean():.3f}, "
                  f"std={cluster_difficulty.std():.3f}, count={np.sum(cluster_mask)}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Dev set:")
    print(f"  L1 distance: {dev_l1_distance:.4f}, L2 distance: {dev_l2_distance:.4f}, Accuracy: {dev_accuracy:.4f}")
    print(f"Val set:")
    print(f"  L1 distance: {val_l1_distance:.4f}, L2 distance: {val_l2_distance:.4f}, Accuracy: {val_accuracy:.4f}")
    print(f"Test set:")
    print(f"  L1 distance: {test_l1_distance:.4f}, L2 distance: {test_l2_distance:.4f}, Accuracy: {test_accuracy:.4f}")
    print(f"\nBin edges used: {bin_edges}")
    
    ### VISUALIZATIONS ###
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Create output directory for figures
    fig_dir = "figures"
    # fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each set
    visualize_clustering_results(
        y_true=y_dev_labels,
        y_pred=dev_predictions,
        difficulty=y_dev_difficulty,
        set_name="dev",
        n_clusters=n_clusters,
        bin_edges=bin_edges,        
        save_path=f"{fig_dir}/dev_clustering_results_{name}.png"
    )
    
    visualize_clustering_results(
        y_true=y_val_labels,
        y_pred=val_predictions,
        difficulty=y_val_difficulty,
        set_name="val",
        n_clusters=n_clusters,
        bin_edges=bin_edges,
        save_path=f"{fig_dir}/val_clustering_results_{name}.png"
    )
    
    visualize_clustering_results(
        y_true=y_test_labels,
        y_pred=test_predictions,
        difficulty=y_test_difficulty,
        set_name="test",
        n_clusters=n_clusters,
        bin_edges=bin_edges,
        save_path=f"{fig_dir}/test_clustering_results_{name}.png"
    )
    
    print(f"Visualizations saved to {fig_dir}/")
    
    ### OPTIONAL: WANDB LOGGING ###
    # wandb.init(entity="329a", project="verification", name="semi_supervised_nb_clustering")
    # wandb.log({
    #     "dev_cluster_accuracy": dev_accuracy,
    #     "n_clusters": n_clusters,
    #     "num_labeled": len(X_dev),
    #     "num_unlabeled": len(X_unlabeled),
    # })
    # wandb.finish()
    
    return model, dev_predictions, val_predictions, test_predictions

    
if __name__ == "__main__":
    hub_name = "wfang11/math500-llama8b-10-10-80"
    fit(hub_name=hub_name)

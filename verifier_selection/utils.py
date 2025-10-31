
from typing import List
from numpy.linalg import norm
from datasets import Dataset
import numpy as np
import datasets
from sklearn.metrics import roc_auc_score

MODEL_PARAMS = {
    "GRM": 8_000_000_000,
    "ArmorRM": 8_000_000_000,
    "URM": 8_000_000_000,
    "QRM": 8_000_000_000,
    "GPM": 8_000_000_000,
    "GRMLlama32": 3_000_000_000,
    "OffsetBias": 8_000_000_000,
    "GRMGemma": 2_000_000_000,
    "Skyworks": 8_000_000_000,
    "SkyworksGemma": 27_000_000_000,
    "QRMGemma": 27_000_000_000,
    "LDLRewardGemma": 27_000_000_000,
    "QwenPRM": 7_000_000_000,
    "Qwen72B": 72_000_000_000,
    "Qwen72BPRM": 72_000_000_000,
    "EurusPRMStage1": 7_000_000_000,
    "EurusPRMStage2": 7_000_000_000,
    "InternLM2RewardModel": 20_000_000_000,
    "InternLM2Reward7B": 7_000_000_000,
    "DecisionTreeReward8B": 8_000_000_000,
    "DecisionTreeReward27B": 27_000_000_000,
    "INFORM": 70_000_000_000,
    "DeepSeekLlama70B": 70_000_000_000,
    "DeepSeekQwen32B": 32_000_000_000,
    "SkyT1": 32_000_000_000,
    "Llama-3.3-70B-Instruct": 70_000_000_000,
    "Meta-Llama-3.1-405B-Instruct-quantized.w8a16": 405_000_000_000,
    "Qwen/Qwen2.5-72B-Instruct": 72_000_000_000,
    "QwQ-32B": 32_000_000_000,
    "WizardLM-2-8x22B": 176_000_000_000,
    "Mixtral-8x22B-Instruct-v0.1": 176_000_000_000,
    "DeepSeekLlama8B": 8_000_000_000,
    "DeepSeekQwen7B": 7_000_000_000,
    "Llama-3.1-8B-Instruct": 8_000_000_000,
    "Gemma-3-12B-Instruct": 12_000_000_000,
    "Gemma-3-4B-Instruct": 4_000_000_000,
    "Phi-4-4B-Instruct": 4_000_000_000,
    "Qwen-2.5-7B-Instruct": 7_000_000_000,
    "Qwen-2.5-Math-7B-Instruct": 7_000_000_000,
    "Mistral-7B-Instruct-v0.2": 7_000_000_000,
}

def load_dataset(dataset_name="hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1") -> datasets.Dataset:
    dataset = datasets.load_dataset(dataset_name)["data"]
    return dataset


def dev_test_split(dataset: Dataset):
    splits = dataset.train_test_split(test_size=0.8, seed=422)
    dev_ds, test_ds = splits["train"], splits["test"]
    return dev_ds, test_ds


def extract_scores_matrix(dataset: Dataset) -> np.ndarray:
    # Extract all verifier score columns
    verifier_columns = [col for col in dataset[0].keys() if (col.endswith('_scores') or col.endswith('_verdicts')) and 'weaver' not in col]
    print(f"Number of verifiers: {len(verifier_columns)}")

    # Create a matrix to store all verifier scores across all problems
    # Shape: (num_problems * K, num_verifiers)
    all_scores = []
    verifier_names = []

    for col in verifier_columns:
        scores_for_verifier = []
        for problem in dataset:
            scores_for_verifier.extend(problem[col])
        all_scores.append(scores_for_verifier)
        verifier_names.append(col)

    scores_matrix = np.array(all_scores).T
    return scores_matrix, verifier_names


def utilities_dict(scores_matrix: np.ndarray, dataset: Dataset, verifier_names: List[str]) -> dict[str, float]:
    ## Model Selection: Balancing Diversity vs Utility

    # First, let's create ground truth labels for utility assessment
    # We'll use the answer_correct labels as our target
    ground_truth = []
    for problem in dataset:
        ground_truth.extend(problem['answer_correct'])

    ground_truth = np.array(ground_truth)
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Overall accuracy: {np.mean(ground_truth):.3f}")

    # Calculate individual verifier utility (AUC-ROC)
    individual_utilities = []
    for i in range(len(verifier_names)):
        try:
            auc = roc_auc_score(ground_truth, scores_matrix[:, i])
            individual_utilities.append(auc)
        except:
            individual_utilities.append(0.5)  # Default for perfect correlation

    return individual_utilities


def similarities_dict(scores_matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pearson": np.corrcoef(scores_matrix, rowvar=False),
        "cosine": np.dot(scores_matrix.T, scores_matrix) / (norm(scores_matrix, axis=0, keepdims=True) * norm(scores_matrix, axis=0, keepdims=True).T),
        "co_agreement": (scores_matrix[:, :, None] == scores_matrix[:, None, :]).mean(axis=0),
    }

def costs_dict(verifier_names: List[str]) -> dict[str, float]:
    param_counts = []
    missing = []
    for name in verifier_names:
        if name in MODEL_PARAMS:
            param_counts.append(float(MODEL_PARAMS[name]))
        elif name.split('_')[0] in MODEL_PARAMS:
            param_counts.append(float(MODEL_PARAMS[name.split('_')[0]]))
        else:
            param_counts.append(0.0)
            print(name)
            missing.append(name)

    print(f"Built parameter count vector. Missing: {len(missing)} of {len(verifier_names)}")
    return param_counts


def greedy_select(
    utilities: np.ndarray,
    similarity_matrix: np.ndarray,
    verifier_names: List[str],
    k: int | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    param_counts: np.ndarray | None = None,
    agg: str = "max",
    start_index: int | None = None
) -> dict:

    n = len(utilities)
    k = n if k is None else min(k, n)
    if param_counts is None:
        costs = np.zeros(n, dtype=float)
    else:
        param_counts_vec = np.array(param_counts, dtype=float)

        # Normalize costs into [0,1] to make gamma interpretable
        if np.nanmax(param_counts_vec) > 0:
            costs = param_counts_vec / np.nanmax(param_counts_vec)
        else:
            costs = np.zeros_like(param_counts_vec)
    # costs = np.zeros(n, dtype=float) if costs is None else costs.astype(float)

    selected: List[int] = []
    remaining: set[int] = set(range(n))

    if start_index is None:
        # start with the best utility, penalized by cost
        start_index = int(np.nanargmax(utilities - gamma * costs))
    selected.append(start_index)
    remaining.remove(start_index)

    def set_similarity(idx: int, chosen: List[int]) -> float:
        if len(chosen) == 0:
            return 0.0
        if agg == "max":
            return float(np.nanmax(similarity_matrix[idx, chosen]))
        else:
            return float(np.nanmean(similarity_matrix[idx, chosen]))

    step_scores: List[float] = [float(alpha * utilities[start_index] - gamma * costs[start_index])]

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -1e18
        for j in list(remaining):
            sim_pen = set_similarity(j, selected)
            cost_pen = costs[j]
            score = alpha * float(utilities[j]) - beta * sim_pen - gamma * cost_pen
            if score > best_score:
                best_score = score
                best_idx = j
        selected.append(best_idx)
        remaining.remove(best_idx)
        step_scores.append(float(best_score))

    return {
        "order": selected,
        "step_scores": step_scores,
        "verifiers": [verifier_names[i] for i in selected],
        "param_counts": [param_counts[i] / 1E9 for i in selected],
    }
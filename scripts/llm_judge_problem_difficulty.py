"""
Load MATH-500 dataset, use few-shot prompt to judge difficulty (1-5) via Together API,
calculate IRR with ground truth, and save to HuggingFace dataset.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
import time

# Add generation directory to path to import generate_together
script_dir = Path(__file__).parent
generation_dir = script_dir.parent / "generation"
sys.path.insert(0, str(generation_dir))
from utils import generate_together


def load_prompt_template():
    """Load the few-shot prompt template from sampled_difficulty_examples.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, "scratchwork", "sampled_difficulty_examples.json")
    
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    
    return data['few_shot_prompt_template']


def extract_difficulty(response: str) -> int:
    """Extract difficulty rating (1-5) from LLM response."""
    response_lower = response.lower()
    
    # First, find all numbers 1-5 in the response
    all_matches = list(re.finditer(r'\b([1-5])\b', response_lower))
    
    # If only one number 1-5 found, use it
    if len(all_matches) == 1:
        return int(all_matches[0].group(1))
    
    # If multiple numbers found, try to find the one closest to "difficulty"
    if len(all_matches) > 1:
        # Find all occurrences of "difficulty" (case-insensitive)
        difficulty_keywords = ['difficulty', 'rating', 'level']
        difficulty_positions = []
        
        for keyword in difficulty_keywords:
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', response_lower):
                difficulty_positions.append(match.start())
        
        if difficulty_positions:
            # Find the number closest to any "difficulty" keyword
            min_distance = float('inf')
            best_match = None
            
            for num_match in all_matches:
                num_pos = num_match.start()
                # Calculate minimum distance to any "difficulty" keyword
                min_dist_to_keyword = min(abs(num_pos - diff_pos) for diff_pos in difficulty_positions)
                
                if min_dist_to_keyword < min_distance:
                    min_distance = min_dist_to_keyword
                    best_match = num_match
            
            if best_match:
                return int(best_match.group(1))
        
        # If we found multiple numbers but couldn't find "difficulty", 
        # look for patterns like "difficulty is X" or "rating: X"
        for num_match in all_matches:
            # Check context around the number for difficulty-related words
            start = max(0, num_match.start() - 50)
            end = min(len(response_lower), num_match.end() + 50)
            context = response_lower[start:end]
            
            if any(kw in context for kw in ['difficulty', 'rating', 'level', 'rate']):
                return int(num_match.group(1))
        
        # Fallback: use the last number found (often the final answer)
        return int(all_matches[-1].group(1))
    
    # If no numbers 1-5 found, try to find any number and clamp to 1-5
    match = re.search(r'\d+', response.strip())
    if match:
        rating = int(match.group(0))
        clamped = max(1, min(5, rating))
        # Only return if it's actually in range 1-5
        if clamped == rating:
            return clamped
    
    return None


def judge_difficulty(problem_text: str, prompt_template: str, model: str, use_system_message: bool = True) -> int:
    """Call LLM to judge problem difficulty using Together API."""
    prompt_template = prompt_template.replace("{problem_text}", problem_text)
    
    if use_system_message:
        # Split into system and user messages for better instruction following
        system_msg = """You are an expert at evaluating the difficulty of mathematical competition problems. 
Your task is to rate difficulty on a scale from 1 to 5. Use the FULL range - do not be conservative.
- Level 1: Very Easy (basic arithmetic, simple algebra)
- Level 2: Easy (standard techniques, minimal complexity)
- Level 3: Medium (requires insight or multiple steps)
- Level 4: Hard (advanced techniques or significant insight needed)
- Level 5: Very Hard (extremely challenging, requires deep expertise)

Respond with ONLY a single digit (1, 2, 3, 4, or 5)."""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_template}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt_template}
        ]
    
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    try:
        response = generate_together(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000  # Just need a single digit
        )
        
        if response is None:
            return None
        
        difficulty = extract_difficulty(response)
        if difficulty is None:
            print(f"Warning: Could not extract difficulty from response: {response[:50]}")
        return difficulty
    except Exception as e:
        print(f"Error calling Together API: {e}")
        return None


def calculate_irr(llm_ratings: list, ground_truth: list):
    """Calculate Inter-Rater Reliability metrics."""
    # Filter out None values
    valid_pairs = [(llm, gt) for llm, gt in zip(llm_ratings, ground_truth) if llm is not None]
    
    if not valid_pairs:
        return None, None, None, None
    
    llm_clean, gt_clean = zip(*valid_pairs)
    
    # Accuracy (exact match)
    accuracy = accuracy_score(gt_clean, llm_clean)
    
    # Pearson correlation
    correlation, p_value = pearsonr(list(llm_clean), list(gt_clean))
    
    # Cohen's kappa
    kappa = cohen_kappa_score(gt_clean, llm_clean)
    
    return accuracy, correlation, kappa, p_value


def sanitize_model_name_for_dataset(model_name: str) -> str:
    """Convert model name to a valid dataset column/dataset name format."""
    # Replace slashes and special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Judge problem difficulty using LLM and calculate IRR with ground truth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        help="Model to use for judging difficulty (default: meta-llama/Llama-3.2-3B-Instruct-Turbo)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process ~30 questions"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=30,
        help="Number of questions to process in test mode (default: 30)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    test_mode = args.test
    test_size = args.test_size
    
    print("=" * 80)
    print("LLM Difficulty Judging")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Test mode: {test_mode}")
    if test_mode:
        print(f"Test size: {test_size} questions")
    print("=" * 80)
    
    print("\nLoading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")['test']
    print(f"Loaded {len(dataset)} problems")
    
    # Limit dataset in test mode
    if test_mode:
        dataset = dataset.select(range(min(test_size, len(dataset))))
        print(f"Test mode: Processing {len(dataset)} problems")
    
    print("Loading prompt template...")
    prompt_template = load_prompt_template()
    
    print("Judging difficulties with LLM...")
    llm_ratings = []
    ground_truth = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Judging difficulties")):
        problem_text = example['problem']
        gt_level = int(example['level'])  # Ground truth level (1-5)
        
        # Judge difficulty
        llm_rating = judge_difficulty(problem_text, prompt_template, model)
        llm_ratings.append(llm_rating)
        ground_truth.append(gt_level)
        
        # Small delay to avoid rate limiting
        if (idx + 1) % 50 == 0:
            time.sleep(1)
    
    print("\nCalculating IRR metrics...")
    accuracy, correlation, kappa, p_value = calculate_irr(llm_ratings, ground_truth)
    
    # Count valid ratings
    valid_count = sum(1 for r in llm_ratings if r is not None)
    invalid_count = len(llm_ratings) - valid_count
    
    print("\n" + "=" * 80)
    print("IRR Results:")
    print("=" * 80)
    print(f"Total problems: {len(ground_truth)}")
    print(f"Valid LLM ratings: {valid_count}")
    print(f"Invalid/Failed ratings: {invalid_count}")
    print(f"\nAccuracy (exact match): {accuracy:.4f}")
    print(f"Pearson Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Distribution comparison
    from collections import Counter
    gt_dist = Counter(ground_truth)
    llm_dist = Counter([r for r in llm_ratings if r is not None])
    
    print("\nGround Truth Distribution:")
    for level in sorted(gt_dist.keys()):
        print(f"  Level {level}: {gt_dist[level]} ({gt_dist[level]/len(ground_truth)*100:.1f}%)")
    
    print("\nLLM Rating Distribution:")
    for level in sorted(llm_dist.keys()):
        print(f"  Level {level}: {llm_dist[level]} ({llm_dist[level]/valid_count*100:.1f}%)")
    
    print("=" * 80)
    
    # Generate column and dataset names based on model
    model_sanitized = sanitize_model_name_for_dataset(model)
    column_name = f"llm_difficulty_fewshot_{model_sanitized}"
    
    # Add LLM difficulty ratings to dataset
    print("\nAdding LLM difficulty ratings to dataset...")
    dataset = dataset.add_column(column_name, llm_ratings)
    
    # Generate dataset name
    if test_mode:
        dataset_name = f"pnsahoo/MATH500_LLM_judge_fewshot_test_{model_sanitized}"
    else:
        dataset_name = f"pnsahoo/MATH500_LLM_judge_fewshot_{model_sanitized}"
    
    print(f"\nPushing dataset to HuggingFace: {dataset_name}")
    print(f"Column name: {column_name}")
    
    try:
        DatasetDict({"test": dataset}).push_to_hub(dataset_name, private=True)
        print(f"Successfully pushed to {dataset_name}")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Saving locally instead...")
        local_path = f"./MATH500_LLM_judge_fewshot_{model_sanitized}"
        if test_mode:
            local_path += "_test"
        DatasetDict({"test": dataset}).save_to_disk(local_path)
        print(f"Saved locally to {local_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()

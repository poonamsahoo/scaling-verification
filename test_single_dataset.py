#!/usr/bin/env python3
"""
Test script to run grid search pipeline on a single dataset.
Use this to verify everything works before running all 18 datasets.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "verifier_selection"))
sys.path.insert(0, str(Path(__file__).parent / "selection"))

from verifier_selection.run_verifier_selection import run_verifier_selection
from selection.run_weaver_evaluation import run_weaver_evaluation


def test_pipeline(dataset_path: str, dataset_name: str, output_dir: str = "test_results"):
    """
    Test the complete pipeline on a single dataset.
    
    Args:
        dataset_path: HuggingFace dataset path
        dataset_name: Name for organizing results
        output_dir: Directory to save test results
    """
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TESTING GRID SEARCH PIPELINE")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Test with reduced hyperparameter space
    k_values = [5, 10]  # Instead of [5, 10, 15]
    beta_values = [0.5, 1.0]  # Instead of [0.25, 0.5, 1.0]
    gamma_values = [0.5, 1.0]  # Instead of [0.25, 0.5, 1.0]
    
    print("\n" + "="*80)
    print("STAGE 1: VERIFIER SELECTION (Test Configuration)")
    print("="*80)
    print(f"k values: {k_values}")
    print(f"beta values: {beta_values}")
    print(f"gamma values: {gamma_values}")
    print(f"Total configurations: {len(k_values) * len(beta_values) * len(gamma_values)}")
    print("="*80 + "\n")
    
    try:
        # Stage 1: Verifier Selection
        results_df, csv_path = run_verifier_selection(
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            k_values=k_values,
            alpha=1.0,
            beta_values=beta_values,
            gamma_values=gamma_values,
            wandb_enabled=False,  # Disable WandB for testing
            save_plots=True
        )
        
        print("\n✅ Stage 1 Complete!")
        print(f"   Results saved to: {csv_path}")
        print(f"   Configurations tested: {len(results_df)}")
        
        # Stage 2: Test Weaver on best configuration
        print("\n" + "="*80)
        print("STAGE 2: WEAVER EVALUATION (Best Configuration Only)")
        print("="*80 + "\n")
        
        # Get best configuration by greedy score
        best_idx = results_df['total_greedy_score'].idxmax()
        best_row = results_df.loc[best_idx]
        
        print(f"Testing with best configuration:")
        print(f"  k={best_row['k']}, beta={best_row['beta']}, gamma={best_row['gamma']}")
        print(f"  Greedy score: {best_row['total_greedy_score']:.3f}")
        
        # Parse verifiers
        verifier_names = [v.strip() for v in best_row['selected_verifiers'].split(',')]
        print(f"  Using {len(verifier_names)} verifiers")
        
        df_train, df_test, summary_path = run_weaver_evaluation(
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            verifier_subset=verifier_names,
            k=int(best_row['k']),
            alpha=best_row['alpha'],
            beta=best_row['beta'],
            gamma=best_row['gamma'],
            wandb_enabled=False  # Disable WandB for testing
        )
        
        print("\n✅ Stage 2 Complete!")
        print(f"   Summary saved to: {summary_path}")
        print(f"   Train accuracy: {df_train['top1_positive'].mean():.3f}")
        print(f"   Test accuracy: {df_test['top1_positive'].mean():.3f}")
        
        # Summary
        print("\n" + "="*80)
        print("TEST PIPELINE SUCCESSFUL!")
        print("="*80)
        print(f"\nResults saved in: {output_dir}")
        print("\nFiles created:")
        print("  - verifier_selection_results.csv")
        print("  - weaver_train_*.csv")
        print("  - weaver_test_*.csv")
        print("  - weaver_summary_*.csv")
        print("  - plots/ (similarity heatmaps and step scores)")
        print("\nYou can now run the full pipeline with:")
        print("  python run_all_datasets_grid_search.py --config datasets_config.yaml")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test grid search pipeline on a single dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with MATH500 70B
  python test_single_dataset.py \\
      --dataset wfang11/math500-llama70b-1-5-94-dev \\
      --name math500-llama70b-test
  
  # Test with GPQA 8B
  python test_single_dataset.py \\
      --dataset wfang11/gpqa-llama8b-1-5-94-dev \\
      --name gpqa-llama8b-test
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset path (e.g., wfang11/math500-llama70b-1-5-94-dev)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for test results directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Base directory for test results (default: test_results)"
    )
    
    args = parser.parse_args()
    
    success = test_pipeline(
        dataset_path=args.dataset,
        dataset_name=args.name,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


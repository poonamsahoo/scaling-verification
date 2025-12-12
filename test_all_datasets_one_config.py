#!/usr/bin/env python3
"""
Quick test script that runs ONE hyperparameter config per dataset
to verify the pipeline works before running the full grid search.
"""

import sys
import yaml
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from verifier_selection.run_verifier_selection import run_verifier_selection
from selection.run_weaver_evaluation import run_weaver_evaluation

def main():
    # Load config
    config_path = Path(__file__).parent / "datasets_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    datasets = config['datasets']
    
    # Test with just ONE hyperparameter config
    k_values = [10]
    alpha = 1.0
    beta_values = [1.0]
    gamma_values = [0.5]
    
    print("=" * 80)
    print("QUICK TEST: Running ONE config per dataset")
    print(f"Test hyperparams: k=10, α=1.0, β=1.0, γ=0.5")
    print(f"Total datasets: {len(datasets)}")
    print("=" * 80)
    
    results = []
    
    for i, ds in enumerate(datasets):
        dataset_name = ds['name']
        dataset_path = ds['dev']  # Use dev split
        
        print(f"\n{'#' * 80}")
        print(f"# [{i+1}/{len(datasets)}] Testing: {dataset_name}")
        print(f"# Path: {dataset_path}")
        print(f"{'#' * 80}\n")
        
        output_dir = Path("results") / f"test_{dataset_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Verifier Selection
            print("STAGE 1: Verifier Selection...")
            vs_results, _ = run_verifier_selection(
                dataset_path=dataset_path,
                output_dir=str(output_dir),
                k_values=k_values,
                alpha=alpha,
                beta_values=beta_values,
                gamma_values=gamma_values,
                wandb_enabled=False,
                save_plots=False
            )
            
            print(f"  ✅ Stage 1 complete")
            print(f"  Selected verifiers: {vs_results['selected_verifiers'].iloc[0][:80]}...")
            
            # Get selected verifiers for Stage 2
            best_row = vs_results.iloc[0]
            selected_verifiers = [v.strip() for v in best_row['selected_verifiers'].split(',')]
            
            # Stage 2: Weaver Evaluation
            print("\nSTAGE 2: Weaver Evaluation...")
            df_train, df_test, summary_path = run_weaver_evaluation(
                dataset_path=dataset_path,
                output_dir=str(output_dir),
                verifier_subset=selected_verifiers,
                k=10,
                alpha=1.0,
                beta=1.0,
                gamma=0.5,
                wandb_enabled=False
            )
            
            test_acc = df_test['select_accuracy'].iloc[0] if 'select_accuracy' in df_test.columns else df_test.iloc[0, 0]
            print(f"  ✅ Stage 2 complete")
            print(f"  Test Accuracy: {test_acc:.3f}")
            
            results.append({
                'dataset': dataset_name,
                'status': '✅ SUCCESS',
                'test_acc': f"{test_acc:.3f}",
                'num_verifiers': len(selected_verifiers)
            })
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_name,
                'status': '❌ FAILED',
                'error': str(e)[:50]
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if 'SUCCESS' in r['status'])
    
    for r in results:
        if 'test_acc' in r:
            print(f"  {r['status']} {r['dataset']}: acc={r['test_acc']}, verifiers={r['num_verifiers']}")
        else:
            print(f"  {r['status']} {r['dataset']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nTotal: {success_count}/{len(results)} datasets passed")
    
    if success_count == len(results):
        print("\n✅ ALL TESTS PASSED! Safe to run full grid search.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Fix issues before running full grid search.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Aggregate results from multi-dataset grid search.
Combines individual dataset results into master summary files.
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json


class ResultsAggregator:
    """Aggregates grid search results from multiple datasets."""
    
    def __init__(self, results_dir: str):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
    
    def find_dataset_dirs(self) -> List[Path]:
        """Find all dataset subdirectories."""
        dataset_dirs = []
        for path in self.results_dir.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                dataset_dirs.append(path)
        return sorted(dataset_dirs)
    
    def aggregate_verifier_selection(self) -> pd.DataFrame:
        """Aggregate Stage 1 (verifier selection) results."""
        print("\n" + "="*80)
        print("AGGREGATING VERIFIER SELECTION RESULTS (Stage 1)")
        print("="*80)
        
        all_results = []
        dataset_dirs = self.find_dataset_dirs()
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            results_file = dataset_dir / "verifier_selection_results.csv"
            
            if not results_file.exists():
                print(f"⚠️  No verifier selection results for {dataset_name}")
                continue
            
            try:
                df = pd.read_csv(results_file)
                df['dataset_name'] = dataset_name
                all_results.append(df)
                print(f"✅ Loaded {len(df)} configs from {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        if not all_results:
            print("❌ No verifier selection results found")
            return None
        
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns for better readability
        cols = ['dataset_name', 'k', 'alpha', 'beta', 'gamma', 
                'total_greedy_score', 'final_step_score', 
                'num_verifiers', 'total_param_GB',
                'selected_verifiers', 'selected_param_counts_GB']
        combined = combined[[c for c in cols if c in combined.columns] + 
                           [c for c in combined.columns if c not in cols]]
        
        # Save combined results
        output_file = self.results_dir / "verifier_selection_all_datasets.csv"
        combined.to_csv(output_file, index=False)
        
        print(f"\n✅ Combined verifier selection results:")
        print(f"   File: {output_file}")
        print(f"   Total configs: {len(combined)}")
        print(f"   Datasets: {combined['dataset_name'].nunique()}")
        
        return combined
    
    def aggregate_weaver_evaluation(self) -> pd.DataFrame:
        """Aggregate Stage 2 (Weaver evaluation) results."""
        print("\n" + "="*80)
        print("AGGREGATING WEAVER EVALUATION RESULTS (Stage 2)")
        print("="*80)
        
        all_results = []
        dataset_dirs = self.find_dataset_dirs()
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            summary_file = dataset_dir / "weaver_all_configs_summary.csv"
            
            if not summary_file.exists():
                print(f"⚠️  No Weaver results for {dataset_name}")
                continue
            
            try:
                df = pd.read_csv(summary_file)
                all_results.append(df)
                print(f"✅ Loaded {len(df)} configs from {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        if not all_results:
            print("❌ No Weaver evaluation results found")
            return None
        
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns for better readability
        cols = ['dataset_name', 'dataset_type', 'model_size',
                'k', 'alpha', 'beta', 'gamma',
                'test_select_accuracy', 'train_select_accuracy',
                'test_sample_accuracy', 'train_sample_accuracy',
                'num_verifiers', 'greedy_score', 'verifiers']
        combined = combined[[c for c in cols if c in combined.columns] + 
                           [c for c in combined.columns if c not in cols]]
        
        # Save combined results
        output_file = self.results_dir / "weaver_evaluation_all_datasets.csv"
        combined.to_csv(output_file, index=False)
        
        print(f"\n✅ Combined Weaver evaluation results:")
        print(f"   File: {output_file}")
        print(f"   Total configs: {len(combined)}")
        print(f"   Datasets: {combined['dataset_name'].nunique()}")
        
        return combined
    
    def create_best_configs_summary(self, weaver_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary of best configuration per dataset."""
        print("\n" + "="*80)
        print("CREATING BEST CONFIGURATIONS SUMMARY")
        print("="*80)
        
        best_configs = []
        
        for dataset_name in weaver_df['dataset_name'].unique():
            dataset_results = weaver_df[weaver_df['dataset_name'] == dataset_name]
            
            # Find best configuration by test accuracy
            best_idx = dataset_results['test_select_accuracy'].idxmax()
            best_row = dataset_results.loc[best_idx]
            
            best_configs.append({
                'dataset_name': dataset_name,
                'dataset_type': best_row.get('dataset_type', ''),
                'model_size': best_row.get('model_size', ''),
                'best_k': best_row['k'],
                'best_alpha': best_row['alpha'],
                'best_beta': best_row['beta'],
                'best_gamma': best_row['gamma'],
                'best_test_accuracy': best_row['test_select_accuracy'],
                'best_train_accuracy': best_row['train_select_accuracy'],
                'num_verifiers': best_row['num_verifiers'],
                'greedy_score': best_row.get('greedy_score', 0),
                'top_verifiers': ', '.join(best_row['verifiers'].split(', ')[:5]) if isinstance(best_row['verifiers'], str) else ''
            })
            
            print(f"  {dataset_name}: Test Acc = {best_row['test_select_accuracy']:.3f} "
                  f"(k={best_row['k']}, β={best_row['beta']}, γ={best_row['gamma']})")
        
        best_df = pd.DataFrame(best_configs)
        output_file = self.results_dir / "best_configs_per_dataset.csv"
        best_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Best configurations saved:")
        print(f"   File: {output_file}")
        print(f"   Datasets: {len(best_df)}")
        
        return best_df
    
    def analyze_verifier_frequency(self, verifier_df: pd.DataFrame) -> Dict:
        """Analyze which verifiers are selected most frequently."""
        print("\n" + "="*80)
        print("ANALYZING VERIFIER SELECTION FREQUENCY")
        print("="*80)
        
        verifier_counts = {}
        
        for _, row in verifier_df.iterrows():
            if pd.isna(row['selected_verifiers']):
                continue
            
            verifiers = [v.strip() for v in str(row['selected_verifiers']).split(',')]
            for verifier in verifiers:
                if verifier:
                    verifier_counts[verifier] = verifier_counts.get(verifier, 0) + 1
        
        # Sort by frequency
        sorted_verifiers = sorted(verifier_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create DataFrame
        freq_df = pd.DataFrame(sorted_verifiers, columns=['verifier', 'selection_count'])
        freq_df['selection_percentage'] = (freq_df['selection_count'] / len(verifier_df)) * 100
        
        output_file = self.results_dir / "verifier_frequency_analysis.csv"
        freq_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Top 10 most selected verifiers:")
        for i, (verifier, count) in enumerate(sorted_verifiers[:10], 1):
            pct = (count / len(verifier_df)) * 100
            print(f"  {i:2d}. {verifier:40s} ({count:3d} times, {pct:5.1f}%)")
        
        print(f"\n✅ Verifier frequency analysis saved:")
        print(f"   File: {output_file}")
        
        return dict(sorted_verifiers)
    
    def create_summary_statistics(self, weaver_df: pd.DataFrame) -> Dict:
        """Create summary statistics across all datasets."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        stats = {
            'total_configurations': len(weaver_df),
            'num_datasets': weaver_df['dataset_name'].nunique(),
            'mean_test_accuracy': weaver_df['test_select_accuracy'].mean(),
            'std_test_accuracy': weaver_df['test_select_accuracy'].std(),
            'max_test_accuracy': weaver_df['test_select_accuracy'].max(),
            'min_test_accuracy': weaver_df['test_select_accuracy'].min(),
            'mean_num_verifiers': weaver_df['num_verifiers'].mean(),
        }
        
        # Per dataset type stats
        if 'dataset_type' in weaver_df.columns:
            for dtype in weaver_df['dataset_type'].unique():
                if pd.notna(dtype):
                    subset = weaver_df[weaver_df['dataset_type'] == dtype]
                    stats[f'{dtype}_mean_test_acc'] = subset['test_select_accuracy'].mean()
                    stats[f'{dtype}_max_test_acc'] = subset['test_select_accuracy'].max()
        
        # Per model size stats
        if 'model_size' in weaver_df.columns:
            for msize in weaver_df['model_size'].unique():
                if pd.notna(msize):
                    subset = weaver_df[weaver_df['model_size'] == msize]
                    stats[f'{msize}_mean_test_acc'] = subset['test_select_accuracy'].mean()
                    stats[f'{msize}_max_test_acc'] = subset['test_select_accuracy'].max()
        
        # Save statistics
        output_file = self.results_dir / "summary_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  Total configurations: {stats['total_configurations']}")
        print(f"  Datasets processed: {stats['num_datasets']}")
        print(f"  Mean test accuracy: {stats['mean_test_accuracy']:.3f} ± {stats['std_test_accuracy']:.3f}")
        print(f"  Max test accuracy: {stats['max_test_accuracy']:.3f}")
        print(f"  Min test accuracy: {stats['min_test_accuracy']:.3f}")
        
        print(f"\n✅ Summary statistics saved:")
        print(f"   File: {output_file}")
        
        return stats
    
    def run_full_aggregation(self):
        """Run complete aggregation pipeline."""
        print("\n" + "#"*80)
        print("# RESULTS AGGREGATION")
        print(f"# Directory: {self.results_dir}")
        print("#"*80)
        
        # Aggregate verifier selection
        verifier_df = self.aggregate_verifier_selection()
        
        # Aggregate Weaver evaluation
        weaver_df = self.aggregate_weaver_evaluation()
        
        if weaver_df is not None:
            # Create best configs summary
            best_df = self.create_best_configs_summary(weaver_df)
            
            # Analyze verifier frequency
            if verifier_df is not None:
                freq_dict = self.analyze_verifier_frequency(verifier_df)
            
            # Create summary statistics
            stats = self.create_summary_statistics(weaver_df)
        
        print("\n" + "#"*80)
        print("# AGGREGATION COMPLETE")
        print("#"*80)
        print(f"\nOutput files in: {self.results_dir}")
        print("  - verifier_selection_all_datasets.csv")
        print("  - weaver_evaluation_all_datasets.csv")
        print("  - best_configs_per_dataset.csv")
        print("  - verifier_frequency_analysis.csv")
        print("  - summary_statistics.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate multi-dataset grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing dataset results (default: results)"
    )
    
    args = parser.parse_args()
    
    try:
        aggregator = ResultsAggregator(args.results_dir)
        aggregator.run_full_aggregation()
        
        print("\n✅ Aggregation successful!")
        print(f"\nNext step: Generate report")
        print(f"  python generate_report.py --results_dir {args.results_dir}")
        
    except Exception as e:
        print(f"\n❌ Aggregation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


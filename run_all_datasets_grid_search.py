#!/usr/bin/env python3
"""
Master script to run comprehensive grid search across all datasets.
Orchestrates both verifier selection (Stage 1) and Weaver evaluation (Stage 2).
"""
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import traceback

# Add to Python path
sys.path.insert(0, str(Path(__file__).parent / "verifier_selection"))
sys.path.insert(0, str(Path(__file__).parent / "selection"))

from verifier_selection.run_verifier_selection import run_verifier_selection
from selection.run_weaver_evaluation import run_weaver_evaluation


class GridSearchOrchestrator:
    """Orchestrates multi-dataset grid search execution."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.log_file = Path(self.config['execution']['log_file'])
        self.completed_datasets = self._load_completed_datasets()
        
    def _load_config(self):
        """Load YAML configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_completed_datasets(self):
        """Load list of already completed datasets for resume capability."""
        if not self.log_file.exists():
            return set()
        
        completed = set()
        with open(self.log_file, 'r') as f:
            for line in f:
                if "COMPLETED:" in line:
                    dataset_name = line.split("COMPLETED:")[1].strip()
                    completed.add(dataset_name)
        return completed
    
    def _log(self, message: str, print_msg: bool = True):
        """Log message to file and optionally print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_msg)
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def run_stage1_verifier_selection(self, dataset_config: dict):
        """Run Stage 1: Verifier selection for a dataset."""
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['dev']
        
        self._log(f"\n{'='*80}")
        self._log(f"STAGE 1: Verifier Selection - {dataset_name}")
        self._log(f"{'='*80}")
        
        output_dir = self.results_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get hyperparameters
        hyperparams = self.config['hyperparameters']
        k_values = hyperparams['k']
        alpha = hyperparams['alpha'][0]  # Typically fixed
        beta_values = hyperparams['beta']
        gamma_values = hyperparams['gamma']
        
        # WandB config
        wandb_enabled = self.config.get('wandb', {}).get('enabled', True)
        wandb_config = None
        if wandb_enabled:
            wandb_config = {
                "entity": self.config['wandb'].get('entity', '329a'),
                "project": self.config['wandb'].get('project', 'verification'),
                "name": f"verifier_sel_{dataset_name}"
            }
        
        try:
            results_df, csv_path = run_verifier_selection(
                dataset_path=dataset_path,
                output_dir=str(output_dir),
                k_values=k_values,
                alpha=alpha,
                beta_values=beta_values,
                gamma_values=gamma_values,
                wandb_enabled=wandb_enabled,
                wandb_config=wandb_config,
                save_plots=True
            )
            
            self._log(f"✅ Stage 1 complete for {dataset_name}")
            self._log(f"   Results: {csv_path}")
            
            return results_df
            
        except Exception as e:
            self._log(f"❌ Stage 1 FAILED for {dataset_name}: {str(e)}")
            self._log(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def run_stage2_weaver_evaluation(self, dataset_config: dict, stage1_results: pd.DataFrame):
        """Run Stage 2: Weaver evaluation with selected verifiers."""
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['dev']
        
        self._log(f"\n{'='*80}")
        self._log(f"STAGE 2: Weaver Evaluation - {dataset_name}")
        self._log(f"{'='*80}")
        
        output_dir = self.results_dir / dataset_name
        
        # WandB config
        wandb_enabled = self.config.get('wandb', {}).get('enabled', True)
        
        stage2_summaries = []
        
        # Run Weaver for each selected verifier configuration
        for idx, row in stage1_results.iterrows():
            k = int(row['k'])
            alpha = row['alpha']
            beta = row['beta']
            gamma = row['gamma']
            
            # Parse selected verifiers
            verifier_names = [v.strip() for v in row['selected_verifiers'].split(',')]
            
            self._log(f"\n  Configuration {idx+1}/{len(stage1_results)}: k={k}, β={beta}, γ={gamma}")
            self._log(f"  Using {len(verifier_names)} verifiers")
            
            wandb_config = None
            if wandb_enabled:
                wandb_config = {
                    "entity": self.config['wandb'].get('entity', '329a'),
                    "project": self.config['wandb'].get('project', 'verification'),
                    "name": f"weaver_{dataset_name}_k{k}_b{beta}_g{gamma}"
                }
            
            try:
                df_train, df_test, summary_path = run_weaver_evaluation(
                    dataset_path=dataset_path,
                    output_dir=str(output_dir),
                    verifier_subset=verifier_names,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    wandb_enabled=wandb_enabled,
                    wandb_config=wandb_config
                )
                
                # Load summary
                summary = pd.read_csv(summary_path)
                summary['dataset_name'] = dataset_name
                summary['dataset_type'] = dataset_config.get('dataset_type', '')
                summary['model_size'] = dataset_config.get('model_size', '')
                summary['greedy_score'] = row['total_greedy_score']
                
                stage2_summaries.append(summary)
                
                self._log(f"  ✅ Complete: Test Acc = {summary['test_select_accuracy'].values[0]:.3f}")
                
            except Exception as e:
                self._log(f"  ❌ FAILED for config k={k}, β={beta}, γ={gamma}: {str(e)}")
                self._log(f"     Traceback: {traceback.format_exc()}")
                continue
        
        if stage2_summaries:
            # Combine all summaries for this dataset
            combined_summary = pd.concat(stage2_summaries, ignore_index=True)
            summary_file = output_dir / "weaver_all_configs_summary.csv"
            combined_summary.to_csv(summary_file, index=False)
            self._log(f"✅ Stage 2 complete for {dataset_name}")
            self._log(f"   Summary: {summary_file}")
            return combined_summary
        else:
            self._log(f"❌ Stage 2 had no successful runs for {dataset_name}")
            return None
    
    def run_dataset(self, dataset_config: dict):
        """Run both stages for a single dataset."""
        dataset_name = dataset_config['name']
        
        # Check if already completed
        if self.config['execution'].get('resume', True) and dataset_name in self.completed_datasets:
            self._log(f"⏭️  Skipping {dataset_name} (already completed)")
            return None
        
        self._log(f"\n{'#'*80}")
        self._log(f"# PROCESSING DATASET: {dataset_name}")
        self._log(f"# {dataset_config.get('display_name', dataset_name)}")
        self._log(f"{'#'*80}")
        
        # Stage 1: Verifier Selection
        stage1_results = self.run_stage1_verifier_selection(dataset_config)
        if stage1_results is None:
            self._log(f"❌ Skipping Stage 2 for {dataset_name} due to Stage 1 failure")
            return None
        
        # Stage 2: Weaver Evaluation
        stage2_results = self.run_stage2_weaver_evaluation(dataset_config, stage1_results)
        
        # Mark as completed
        self._log(f"COMPLETED: {dataset_name}")
        self.completed_datasets.add(dataset_name)
        
        return stage2_results
    
    def run_all_datasets(self, dataset_filter: list = None):
        """Run grid search on all datasets (or filtered subset)."""
        datasets = self.config['datasets']
        
        # Filter if specified
        if dataset_filter:
            datasets = [d for d in datasets if d['name'] in dataset_filter]
        
        self._log(f"\n{'#'*80}")
        self._log(f"# MULTI-DATASET GRID SEARCH")
        self._log(f"# Total datasets: {len(datasets)}")
        self._log(f"# Config: {self.config_path}")
        self._log(f"{'#'*80}\n")
        
        all_results = []
        
        # Process each dataset with progress bar
        for dataset_config in tqdm(datasets, desc="Processing datasets"):
            result = self.run_dataset(dataset_config)
            if result is not None:
                all_results.append(result)
        
        # Aggregate results
        if all_results:
            self._log(f"\n{'='*80}")
            self._log("AGGREGATING RESULTS")
            self._log(f"{'='*80}")
            
            combined = pd.concat(all_results, ignore_index=True)
            output_path = self.results_dir / "all_datasets_summary.csv"
            combined.to_csv(output_path, index=False)
            
            self._log(f"✅ Combined results saved to: {output_path}")
            self._log(f"   Total configurations: {len(combined)}")
            self._log(f"   Datasets completed: {len(all_results)}")
            
            return combined
        else:
            self._log("❌ No results to aggregate")
            return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive grid search across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 18 datasets
  python run_all_datasets_grid_search.py --config datasets_config.yaml
  
  # Run specific datasets only
  python run_all_datasets_grid_search.py --config datasets_config.yaml \\
      --datasets math500-llama70b gpqa-llama70b
  
  # Resume from previous run
  python run_all_datasets_grid_search.py --config datasets_config.yaml --resume
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="datasets_config.yaml",
        help="Path to configuration YAML file (default: datasets_config.yaml)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=None,
        help="Specific dataset names to process (default: all datasets in config)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already completed datasets (start fresh)"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = GridSearchOrchestrator(args.config)
    
    # Override resume setting if specified
    if args.no_resume:
        orchestrator.config['execution']['resume'] = False
    
    # Run grid search
    try:
        results = orchestrator.run_all_datasets(dataset_filter=args.datasets)
        
        if results is not None:
            print(f"\n{'='*80}")
            print("🎉 GRID SEARCH COMPLETE!")
            print(f"{'='*80}")
            print(f"Results saved to: {orchestrator.results_dir}")
            print(f"Log file: {orchestrator.log_file}")
            print(f"\nNext steps:")
            print(f"  1. Review results: {orchestrator.results_dir / 'all_datasets_summary.csv'}")
            print(f"  2. Generate report: python generate_report.py --results_dir {orchestrator.results_dir}")
            print(f"  3. View WandB: https://wandb.ai/{orchestrator.config['wandb']['entity']}/{orchestrator.config['wandb']['project']}")
        else:
            print("\n❌ Grid search completed with errors. Check log file for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Grid search interrupted by user")
        print(f"Progress saved to: {orchestrator.log_file}")
        print("Run again with --resume to continue from where you left off")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()


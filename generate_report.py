#!/usr/bin/env python3
"""
Generate comprehensive report from multi-dataset grid search results.
Creates markdown report with tables, statistics, and visualizations.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import numpy as np


class ReportGenerator:
    """Generates analysis report from aggregated results."""
    
    def __init__(self, results_dir: str, output_file: str = None):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
        
        self.output_file = Path(output_file) if output_file else self.results_dir / "report.md"
        self.figures_dir = self.results_dir / "report_figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        self.report_lines = []
    
    def add_line(self, line: str = ""):
        """Add line to report."""
        self.report_lines.append(line)
    
    def add_header(self, text: str, level: int = 1):
        """Add markdown header."""
        self.add_line(f"{'#' * level} {text}\n")
    
    def add_table(self, df: pd.DataFrame, max_rows: int = None):
        """Add markdown table from DataFrame."""
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
            self.add_line(f"*Showing top {max_rows} of {len(df)} rows*\n")
        
        self.add_line(df.to_markdown(index=False))
        self.add_line()
    
    def load_data(self):
        """Load all aggregated data files."""
        self.weaver_df = pd.read_csv(self.results_dir / "weaver_evaluation_all_datasets.csv")
        self.best_configs_df = pd.read_csv(self.results_dir / "best_configs_per_dataset.csv")
        
        # Load optional files
        verifier_sel_file = self.results_dir / "verifier_selection_all_datasets.csv"
        if verifier_sel_file.exists():
            self.verifier_df = pd.read_csv(verifier_sel_file)
        else:
            self.verifier_df = None
        
        freq_file = self.results_dir / "verifier_frequency_analysis.csv"
        if freq_file.exists():
            self.freq_df = pd.read_csv(freq_file)
        else:
            self.freq_df = None
        
        stats_file = self.results_dir / "summary_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
    
    def generate_title_page(self):
        """Generate report title and metadata."""
        self.add_header("Multi-Dataset Grid Search Results", level=1)
        self.add_line(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_line(f"**Results Directory:** `{self.results_dir}`")
        self.add_line()
        self.add_line("---")
        self.add_line()
    
    def generate_executive_summary(self):
        """Generate executive summary section."""
        self.add_header("Executive Summary", level=2)
        
        self.add_line(f"- **Total Configurations Tested:** {len(self.weaver_df)}")
        self.add_line(f"- **Datasets Processed:** {self.weaver_df['dataset_name'].nunique()}")
        self.add_line(f"- **Mean Test Accuracy:** {self.weaver_df['test_select_accuracy'].mean():.3f} ± {self.weaver_df['test_select_accuracy'].std():.3f}")
        self.add_line(f"- **Best Overall Accuracy:** {self.weaver_df['test_select_accuracy'].max():.3f}")
        
        best_row = self.weaver_df.loc[self.weaver_df['test_select_accuracy'].idxmax()]
        self.add_line(f"- **Best Configuration:** {best_row['dataset_name']} with k={best_row['k']}, β={best_row['beta']}, γ={best_row['gamma']}")
        self.add_line()
    
    def generate_best_configs_table(self):
        """Generate best configurations table."""
        self.add_header("Best Configuration Per Dataset", level=2)
        
        # Select columns for display
        display_cols = ['dataset_name', 'dataset_type', 'model_size', 
                       'best_test_accuracy', 'best_k', 'best_beta', 'best_gamma',
                       'num_verifiers']
        
        display_df = self.best_configs_df[display_cols].copy()
        display_df = display_df.rename(columns={
            'dataset_name': 'Dataset',
            'dataset_type': 'Type',
            'model_size': 'Model',
            'best_test_accuracy': 'Test Acc',
            'best_k': 'k',
            'best_beta': 'β',
            'best_gamma': 'γ',
            'num_verifiers': '# Verifiers'
        })
        
        # Round numeric columns
        display_df['Test Acc'] = display_df['Test Acc'].round(3)
        
        self.add_table(display_df)
    
    def generate_verifier_frequency_analysis(self):
        """Generate verifier frequency analysis."""
        if self.freq_df is None:
            return
        
        self.add_header("Verifier Selection Frequency", level=2)
        
        self.add_line("### Top 20 Most Frequently Selected Verifiers\n")
        
        top20 = self.freq_df.head(20).copy()
        top20 = top20.rename(columns={
            'verifier': 'Verifier',
            'selection_count': 'Times Selected',
            'selection_percentage': 'Selection %'
        })
        top20['Selection %'] = top20['Selection %'].round(1)
        
        self.add_table(top20)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        top15 = self.freq_df.head(15)
        ax.barh(range(len(top15)), top15['selection_count'])
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15['verifier'])
        ax.set_xlabel('Selection Count')
        ax.set_title('Top 15 Most Frequently Selected Verifiers')
        ax.invert_yaxis()
        plt.tight_layout()
        
        fig_path = self.figures_dir / "verifier_frequency.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.add_line(f"![Verifier Frequency](report_figures/verifier_frequency.png)")
        self.add_line()
    
    def generate_accuracy_comparison_plots(self):
        """Generate accuracy comparison visualizations."""
        self.add_header("Performance Analysis", level=2)
        
        # 1. Box plot by dataset type
        if 'dataset_type' in self.weaver_df.columns:
            self.add_line("### Test Accuracy by Dataset Type\n")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            self.weaver_df.boxplot(column='test_select_accuracy', 
                                   by='dataset_type', ax=ax)
            ax.set_xlabel('Dataset Type')
            ax.set_ylabel('Test Select Accuracy')
            ax.set_title('Test Accuracy Distribution by Dataset Type')
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            
            fig_path = self.figures_dir / "accuracy_by_dataset_type.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.add_line(f"![Accuracy by Dataset Type](report_figures/accuracy_by_dataset_type.png)")
            self.add_line()
        
        # 2. Box plot by model size
        if 'model_size' in self.weaver_df.columns:
            self.add_line("### Test Accuracy by Model Size\n")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            self.weaver_df.boxplot(column='test_select_accuracy', 
                                   by='model_size', ax=ax)
            ax.set_xlabel('Model Size')
            ax.set_ylabel('Test Select Accuracy')
            ax.set_title('Test Accuracy Distribution by Model Size')
            plt.suptitle('')
            plt.tight_layout()
            
            fig_path = self.figures_dir / "accuracy_by_model_size.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.add_line(f"![Accuracy by Model Size](report_figures/accuracy_by_model_size.png)")
            self.add_line()
        
        # 3. Heatmap of k vs beta vs gamma
        self.add_line("### Hyperparameter Performance Heatmap\n")
        
        # Average accuracy for each hyperparameter combination
        hparam_avg = self.weaver_df.groupby(['k', 'beta', 'gamma'])['test_select_accuracy'].mean().reset_index()
        
        # Create separate heatmap for each k value
        for k_val in sorted(hparam_avg['k'].unique()):
            subset = hparam_avg[hparam_avg['k'] == k_val]
            pivot = subset.pivot(index='beta', columns='gamma', values='test_select_accuracy')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       vmin=pivot.min().min(), vmax=pivot.max().max(),
                       ax=ax, cbar_kws={'label': 'Test Accuracy'})
            ax.set_title(f'Average Test Accuracy (k={k_val})')
            ax.set_xlabel('γ (Cost Weight)')
            ax.set_ylabel('β (Similarity Weight)')
            plt.tight_layout()
            
            fig_path = self.figures_dir / f"heatmap_k{k_val}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.add_line(f"![Heatmap k={k_val}](report_figures/heatmap_k{k_val}.png)")
            self.add_line()
    
    def generate_dataset_comparison_plot(self):
        """Generate dataset comparison visualization."""
        self.add_line("### Performance Across All Datasets\n")
        
        # Get best accuracy for each dataset
        best_per_dataset = self.best_configs_df.sort_values('best_test_accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(best_per_dataset)), best_per_dataset['best_test_accuracy'])
        
        # Color by dataset type if available
        if 'dataset_type' in best_per_dataset.columns:
            colors = {'MATH-500': 'blue', 'GPQA': 'green', 'MMLU': 'orange'}
            for i, (_, row) in enumerate(best_per_dataset.iterrows()):
                bars[i].set_color(colors.get(row['dataset_type'], 'gray'))
        
        ax.set_yticks(range(len(best_per_dataset)))
        ax.set_yticklabels(best_per_dataset['dataset_name'])
        ax.set_xlabel('Best Test Select Accuracy')
        ax.set_title('Best Performance Across All Datasets')
        ax.set_xlim(0, 1.0)
        plt.tight_layout()
        
        fig_path = self.figures_dir / "dataset_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.add_line(f"![Dataset Comparison](report_figures/dataset_comparison.png)")
        self.add_line()
    
    def generate_detailed_stats(self):
        """Generate detailed statistics section."""
        self.add_header("Detailed Statistics", level=2)
        
        # Overall stats
        self.add_line("### Overall Performance\n")
        self.add_line(f"- **Mean Test Accuracy:** {self.weaver_df['test_select_accuracy'].mean():.4f}")
        self.add_line(f"- **Std Test Accuracy:** {self.weaver_df['test_select_accuracy'].std():.4f}")
        self.add_line(f"- **Median Test Accuracy:** {self.weaver_df['test_select_accuracy'].median():.4f}")
        self.add_line(f"- **Min Test Accuracy:** {self.weaver_df['test_select_accuracy'].min():.4f}")
        self.add_line(f"- **Max Test Accuracy:** {self.weaver_df['test_select_accuracy'].max():.4f}")
        self.add_line()
        
        # Per dataset type
        if 'dataset_type' in self.weaver_df.columns:
            self.add_line("### Performance by Dataset Type\n")
            for dtype in sorted(self.weaver_df['dataset_type'].dropna().unique()):
                subset = self.weaver_df[self.weaver_df['dataset_type'] == dtype]
                self.add_line(f"**{dtype}:**")
                self.add_line(f"- Mean: {subset['test_select_accuracy'].mean():.4f}")
                self.add_line(f"- Max: {subset['test_select_accuracy'].max():.4f}")
                self.add_line(f"- Configs: {len(subset)}")
                self.add_line()
        
        # Per model size
        if 'model_size' in self.weaver_df.columns:
            self.add_line("### Performance by Model Size\n")
            for msize in sorted(self.weaver_df['model_size'].dropna().unique()):
                subset = self.weaver_df[self.weaver_df['model_size'] == msize]
                self.add_line(f"**{msize}:**")
                self.add_line(f"- Mean: {subset['test_select_accuracy'].mean():.4f}")
                self.add_line(f"- Max: {subset['test_select_accuracy'].max():.4f}")
                self.add_line(f"- Configs: {len(subset)}")
                self.add_line()
    
    def generate_recommendations(self):
        """Generate recommendations section."""
        self.add_header("Recommendations", level=2)
        
        # Find most consistent hyperparameters
        best_k = self.weaver_df.groupby('k')['test_select_accuracy'].mean().idxmax()
        best_beta = self.weaver_df.groupby('beta')['test_select_accuracy'].mean().idxmax()
        best_gamma = self.weaver_df.groupby('gamma')['test_select_accuracy'].mean().idxmax()
        
        self.add_line("### Recommended Hyperparameters (based on average performance)\n")
        self.add_line(f"- **k (number of verifiers):** {best_k}")
        self.add_line(f"- **β (similarity penalty):** {best_beta}")
        self.add_line(f"- **γ (cost penalty):** {best_gamma}")
        self.add_line()
        
        # Most frequently selected verifiers
        if self.freq_df is not None:
            self.add_line("### Most Reliable Verifiers\n")
            self.add_line("Top 5 most frequently selected verifiers:")
            for i, row in self.freq_df.head(5).iterrows():
                self.add_line(f"{i+1}. {row['verifier']} (selected {row['selection_count']} times)")
            self.add_line()
    
    def generate_full_report(self):
        """Generate complete report."""
        print("Generating comprehensive report...")
        
        # Load data
        print("  Loading data...")
        self.load_data()
        
        # Generate sections
        print("  Generating title page...")
        self.generate_title_page()
        
        print("  Generating executive summary...")
        self.generate_executive_summary()
        
        print("  Generating best configs table...")
        self.generate_best_configs_table()
        
        print("  Generating verifier frequency analysis...")
        self.generate_verifier_frequency_analysis()
        
        print("  Generating performance plots...")
        self.generate_accuracy_comparison_plots()
        
        print("  Generating dataset comparison...")
        self.generate_dataset_comparison_plot()
        
        print("  Generating detailed statistics...")
        self.generate_detailed_stats()
        
        print("  Generating recommendations...")
        self.generate_recommendations()
        
        # Write report
        print(f"  Writing report to {self.output_file}...")
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(self.report_lines))
        
        print(f"\n✅ Report generated successfully!")
        print(f"   Report: {self.output_file}")
        print(f"   Figures: {self.figures_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report from grid search results"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing aggregated results (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report file (default: results/report.md)"
    )
    
    args = parser.parse_args()
    
    try:
        generator = ReportGenerator(args.results_dir, args.output)
        generator.generate_full_report()
        
        print(f"\n📊 Report ready! View with:")
        print(f"   cat {generator.output_file}")
        print(f"   # or open in browser/editor")
        
        return 0
    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())


"""
Visualization module for facial similarity analysis.

Creates publication-quality plots and figures for the blog post.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class SimilarityVisualizer:
    """Creates visualizations for similarity analysis."""

    def __init__(self, results_file: str = "results/statistical_analysis.json"):
        """
        Initialize visualizer.

        Args:
            results_file: Path to statistical analysis results
        """
        self.results_file = results_file
        self.results = self._load_results()
        self.output_dir = Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_results(self) -> Dict:
        """Load analysis results."""
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def plot_distribution_comparison(self):
        """Create histogram comparing couple vs random pair similarities."""
        couple_sims = self.results["raw_data"]["couple_similarities"]
        random_sims = self.results["raw_data"]["random_similarities"]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histograms
        ax.hist(random_sims, bins=30, alpha=0.6, label='Random Pairs',
                color='#3498db', edgecolor='black')
        ax.hist(couple_sims, bins=30, alpha=0.6, label='Couples',
                color='#e74c3c', edgecolor='black')

        # Add mean lines
        ax.axvline(np.mean(random_sims), color='#2874a6', linestyle='--',
                   linewidth=2, label=f'Random Mean: {np.mean(random_sims):.3f}')
        ax.axvline(np.mean(couple_sims), color='#c0392b', linestyle='--',
                   linewidth=2, label=f'Couple Mean: {np.mean(couple_sims):.3f}')

        ax.set_xlabel('Facial Similarity (Cosine Similarity)', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title('Distribution of Facial Similarity: Couples vs Random Pairs',
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution comparison to {output_path}")
        plt.close()

    def plot_violin_comparison(self):
        """Create violin plot comparing distributions."""
        couple_sims = self.results["raw_data"]["couple_similarities"]
        random_sims = self.results["raw_data"]["random_similarities"]

        # Prepare data
        data = {
            'Similarity': couple_sims + random_sims,
            'Group': ['Couples'] * len(couple_sims) + ['Random Pairs'] * len(random_sims)
        }

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create violin plot
        parts = ax.violinplot(
            [random_sims, couple_sims],
            positions=[1, 2],
            showmeans=True,
            showextrema=True,
            widths=0.7
        )

        # Customize colors
        colors = ['#3498db', '#e74c3c']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Random Pairs', 'Couples'], fontsize=14)
        ax.set_ylabel('Facial Similarity (Cosine Similarity)', fontsize=14)
        ax.set_title('Facial Similarity Distribution Comparison',
                     fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "violin_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved violin plot to {output_path}")
        plt.close()

    def plot_box_comparison(self):
        """Create box plot comparison."""
        couple_sims = self.results["raw_data"]["couple_similarities"]
        random_sims = self.results["raw_data"]["random_similarities"]

        fig, ax = plt.subplots(figsize=(10, 8))

        box_data = [random_sims, couple_sims]
        bp = ax.boxplot(box_data, labels=['Random Pairs', 'Couples'],
                        patch_artist=True, widths=0.6)

        # Customize colors
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add sample size annotations
        ax.text(1, np.max(random_sims) * 0.95, f'n={len(random_sims)}',
                ha='center', fontsize=11)
        ax.text(2, np.max(couple_sims) * 0.95, f'n={len(couple_sims)}',
                ha='center', fontsize=11)

        ax.set_ylabel('Facial Similarity (Cosine Similarity)', fontsize=14)
        ax.set_title('Facial Similarity: Box Plot Comparison',
                     fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "box_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved box plot to {output_path}")
        plt.close()

    def plot_statistical_summary(self):
        """Create statistical summary visualization."""
        desc = self.results["descriptive_statistics"]
        t_test = self.results["t_test"]
        cohens_d = self.results["cohens_d"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Mean comparison
        groups = ['Random Pairs', 'Couples']
        means = [desc['random_pairs']['mean'], desc['couples']['mean']]
        stds = [desc['random_pairs']['std'], desc['couples']['std']]

        ax1.bar(groups, means, yerr=stds, capsize=10,
                color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Similarity', fontsize=12)
        ax1.set_title('Mean Facial Similarity Comparison', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')

        # Add values on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(i, mean + std + 0.01, f'{mean:.3f}\n±{std:.3f}',
                    ha='center', fontsize=10)

        # 2. Sample sizes
        ns = [desc['random_pairs']['n'], desc['couples']['n']]
        ax2.bar(groups, ns, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Sample Size', fontsize=12)
        ax2.set_title('Sample Sizes', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')

        # Add values on bars
        for i, n in enumerate(ns):
            ax2.text(i, n + 1, str(n), ha='center', fontsize=11, fontweight='bold')

        # 3. Statistical significance
        tests = ['T-Test', 'Mann-Whitney', 'Permutation']
        p_values = [
            t_test['p_value'],
            self.results['mann_whitney_test']['p_value'],
            self.results['permutation_test']['p_value']
        ]

        colors_sig = ['#27ae60' if p < 0.05 else '#e67e22' for p in p_values]
        ax3.bar(tests, p_values, color=colors_sig, alpha=0.7, edgecolor='black')
        ax3.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax3.set_ylabel('P-Value', fontsize=12)
        ax3.set_title('Statistical Test Results', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        ax3.set_ylim([0, max(p_values) * 1.2])

        # Add values on bars
        for i, p in enumerate(p_values):
            ax3.text(i, p + max(p_values) * 0.02, f'{p:.4f}',
                    ha='center', fontsize=9)

        # 4. Effect size interpretation
        ax4.axis('off')
        effect_text = f"""
        Effect Size (Cohen's d): {cohens_d:.3f}

        Interpretation:
        • |d| < 0.2: Small effect
        • |d| = 0.2-0.5: Medium effect
        • |d| = 0.5-0.8: Large effect
        • |d| > 0.8: Very large effect

        Result: {'SIGNIFICANT' if t_test['p_value'] < 0.05 else 'NOT SIGNIFICANT'}

        {t_test['interpretation']}
        """
        ax4.text(0.1, 0.5, effect_text, fontsize=11,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / "statistical_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved statistical summary to {output_path}")
        plt.close()

    def create_all_visualizations(self):
        """Generate all visualizations."""
        logger.info("Creating distribution comparison...")
        self.plot_distribution_comparison()

        logger.info("Creating violin plot...")
        self.plot_violin_comparison()

        logger.info("Creating box plot...")
        self.plot_box_comparison()

        logger.info("Creating statistical summary...")
        self.plot_statistical_summary()

        logger.info(f"\nAll visualizations saved to {self.output_dir}/")


def main():
    """Generate all visualizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create visualizations for similarity analysis"
    )
    parser.add_argument(
        "--results-file",
        default="results/statistical_analysis.json",
        help="Path to statistical analysis results"
    )

    args = parser.parse_args()

    visualizer = SimilarityVisualizer(args.results_file)
    visualizer.create_all_visualizations()

    print("\n" + "="*60)
    print("Visualizations created successfully!")
    print(f"Check the {visualizer.output_dir}/ directory for plots")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

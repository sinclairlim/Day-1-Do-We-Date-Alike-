"""
Statistical analysis of facial similarity in couples.

This module performs hypothesis testing to determine if couples
are more similar than random pairs.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Analyzes facial similarity patterns in couples."""

    def __init__(self, embeddings_file: str = "data/processed/embeddings_results.json"):
        """
        Initialize analyzer with embeddings data.

        Args:
            embeddings_file: Path to embeddings results JSON
        """
        self.embeddings_file = embeddings_file
        self.data = self._load_data()
        self.couple_similarities = [item["similarity"] for item in self.data]

    def _load_data(self) -> List[Dict]:
        """Load embeddings results."""
        with open(self.embeddings_file, 'r') as f:
            return json.load(f)

    def generate_random_pairs(self, n_pairs: int = None) -> List[float]:
        """
        Generate random pair similarities as baseline.

        Args:
            n_pairs: Number of random pairs (default: same as couple count)

        Returns:
            List of similarity scores for random pairs
        """
        if n_pairs is None:
            n_pairs = len(self.data)

        # Load all embeddings
        all_embeddings = []
        for item in self.data:
            emb1 = np.load(item["embedding1_path"])
            emb2 = np.load(item["embedding2_path"])
            all_embeddings.extend([emb1, emb2])

        # Generate random pairs
        random_similarities = []
        for _ in range(n_pairs):
            idx1, idx2 = np.random.choice(len(all_embeddings), 2, replace=False)
            emb1, emb2 = all_embeddings[idx1], all_embeddings[idx2]

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            random_similarities.append(float(similarity))

        return random_similarities

    def t_test(self, random_similarities: List[float]) -> Dict:
        """
        Perform independent t-test.

        Args:
            random_similarities: Similarities for random pairs

        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_ind(
            self.couple_similarities,
            random_similarities
        )

        return {
            "test": "independent_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "interpretation": "Couples are significantly more similar" if p_value < 0.05 else "No significant difference"
        }

    def mann_whitney_test(self, random_similarities: List[float]) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).

        Args:
            random_similarities: Similarities for random pairs

        Returns:
            Dictionary with test results
        """
        u_stat, p_value = stats.mannwhitneyu(
            self.couple_similarities,
            random_similarities,
            alternative='greater'
        )

        return {
            "test": "mann_whitney_u",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "interpretation": "Couples are significantly more similar" if p_value < 0.05 else "No significant difference"
        }

    def effect_size_cohens_d(self, random_similarities: List[float]) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            random_similarities: Similarities for random pairs

        Returns:
            Cohen's d value
        """
        mean1 = np.mean(self.couple_similarities)
        mean2 = np.mean(random_similarities)

        n1, n2 = len(self.couple_similarities), len(random_similarities)
        var1 = np.var(self.couple_similarities, ddof=1)
        var2 = np.var(random_similarities, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Handle case where pooled_std is 0 or very small (data nearly identical)
        if pooled_std < 1e-10:
            cohens_d = 0.0  # No meaningful difference if no variance
        else:
            cohens_d = (mean1 - mean2) / pooled_std

        return float(cohens_d)

    def permutation_test(
        self,
        random_similarities: List[float],
        n_permutations: int = 10000
    ) -> Dict:
        """
        Perform permutation test.

        Args:
            random_similarities: Similarities for random pairs
            n_permutations: Number of permutations

        Returns:
            Dictionary with test results
        """
        observed_diff = np.mean(self.couple_similarities) - np.mean(random_similarities)

        combined = np.array(self.couple_similarities + random_similarities)
        n_couples = len(self.couple_similarities)

        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_couples = combined[:n_couples]
            perm_random = combined[n_couples:]

            perm_diff = np.mean(perm_couples) - np.mean(perm_random)

            if perm_diff >= observed_diff:
                count += 1

        p_value = count / n_permutations

        return {
            "test": "permutation_test",
            "observed_difference": float(observed_diff),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "significant": bool(p_value < 0.05),
            "interpretation": "Couples are significantly more similar" if p_value < 0.05 else "No significant difference"
        }

    def descriptive_statistics(self, random_similarities: List[float]) -> Dict:
        """
        Calculate descriptive statistics.

        Args:
            random_similarities: Similarities for random pairs

        Returns:
            Dictionary with descriptive stats
        """
        return {
            "couples": {
                "n": len(self.couple_similarities),
                "mean": float(np.mean(self.couple_similarities)),
                "median": float(np.median(self.couple_similarities)),
                "std": float(np.std(self.couple_similarities)),
                "min": float(np.min(self.couple_similarities)),
                "max": float(np.max(self.couple_similarities)),
                "q25": float(np.percentile(self.couple_similarities, 25)),
                "q75": float(np.percentile(self.couple_similarities, 75))
            },
            "random_pairs": {
                "n": len(random_similarities),
                "mean": float(np.mean(random_similarities)),
                "median": float(np.median(random_similarities)),
                "std": float(np.std(random_similarities)),
                "min": float(np.min(random_similarities)),
                "max": float(np.max(random_similarities)),
                "q25": float(np.percentile(random_similarities, 25)),
                "q75": float(np.percentile(random_similarities, 75))
            }
        }

    def run_full_analysis(self, n_random_pairs: int = None) -> Dict:
        """
        Run complete statistical analysis.

        Args:
            n_random_pairs: Number of random pairs to generate

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Generating random pair similarities...")
        random_similarities = self.generate_random_pairs(n_random_pairs)

        logger.info("Calculating descriptive statistics...")
        descriptive = self.descriptive_statistics(random_similarities)

        logger.info("Performing t-test...")
        t_test_results = self.t_test(random_similarities)

        logger.info("Performing Mann-Whitney U test...")
        mann_whitney_results = self.mann_whitney_test(random_similarities)

        logger.info("Calculating effect size...")
        cohens_d = self.effect_size_cohens_d(random_similarities)

        logger.info("Performing permutation test...")
        perm_test_results = self.permutation_test(random_similarities)

        results = {
            "descriptive_statistics": descriptive,
            "t_test": t_test_results,
            "mann_whitney_test": mann_whitney_results,
            "cohens_d": cohens_d,
            "permutation_test": perm_test_results,
            "raw_data": {
                "couple_similarities": self.couple_similarities,
                "random_similarities": random_similarities
            }
        }

        return results


def main():
    """Run statistical analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze facial similarity in couples"
    )
    parser.add_argument(
        "--embeddings-file",
        default="data/processed/embeddings_results.json",
        help="Path to embeddings results JSON"
    )
    parser.add_argument(
        "--output-file",
        default="results/statistical_analysis.json",
        help="Path to save analysis results"
    )
    parser.add_argument(
        "--n-random-pairs",
        type=int,
        default=None,
        help="Number of random pairs to generate (default: same as couple count)"
    )

    args = parser.parse_args()

    # Create results directory
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = SimilarityAnalyzer(args.embeddings_file)
    results = analyzer.run_full_analysis(args.n_random_pairs)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)

    desc = results["descriptive_statistics"]
    print(f"\nCouples (n={desc['couples']['n']}):")
    print(f"  Mean similarity: {desc['couples']['mean']:.4f}")
    print(f"  Std deviation: {desc['couples']['std']:.4f}")

    print(f"\nRandom Pairs (n={desc['random_pairs']['n']}):")
    print(f"  Mean similarity: {desc['random_pairs']['mean']:.4f}")
    print(f"  Std deviation: {desc['random_pairs']['std']:.4f}")

    print(f"\nT-Test:")
    print(f"  t-statistic: {results['t_test']['t_statistic']:.4f}")
    print(f"  p-value: {results['t_test']['p_value']:.4f}")
    print(f"  {results['t_test']['interpretation']}")

    print(f"\nEffect Size (Cohen's d): {results['cohens_d']:.4f}")

    print(f"\nPermutation Test:")
    print(f"  p-value: {results['permutation_test']['p_value']:.4f}")
    print(f"  {results['permutation_test']['interpretation']}")

    print("\n" + "="*60)
    print(f"Full results saved to {args.output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

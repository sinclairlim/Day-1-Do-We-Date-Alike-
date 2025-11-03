#!/usr/bin/env python3
"""
Main pipeline script to run the entire analysis.

This script orchestrates the complete analysis pipeline:
1. Extract facial embeddings from couple images
2. Perform statistical analysis
3. Generate visualizations
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.facial_analysis.extract_embeddings import process_all_couples
from src.statistical_analysis import SimilarityAnalyzer
from src.visualization import SimilarityVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete facial similarity analysis pipeline"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip embedding extraction (use existing embeddings)"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip statistical analysis (use existing results)"
    )
    parser.add_argument(
        "--detector",
        default="opencv",
        choices=["opencv", "mtcnn", "dlib"],
        help="Face detection method"
    )
    parser.add_argument(
        "--model",
        default="facenet",
        choices=["facenet", "vggface", "arcface"],
        help="Embedding model"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DO WE DATE ALIKE? - Analysis Pipeline")
    print("="*70 + "\n")

    # Step 1: Extract embeddings
    if not args.skip_extraction:
        logger.info("STEP 1: Extracting facial embeddings...")
        print("\n" + "-"*70)
        print("STEP 1: Extracting Facial Embeddings")
        print("-"*70)

        try:
            results = process_all_couples(
                data_dir="data/raw",
                processed_dir="data/processed",
                detector_method=args.detector,
                model_name=args.model
            )
            print(f"\n✓ Successfully processed {len(results)} couples")
        except Exception as e:
            logger.error(f"Error in embedding extraction: {e}")
            print(f"\n✗ Error: {e}")
            print("Make sure you have couple images in the data/raw/ directory")
            return 1
    else:
        logger.info("Skipping embedding extraction...")
        print("\nSkipping embedding extraction (using existing data)...")

    # Step 2: Statistical analysis
    if not args.skip_analysis:
        logger.info("STEP 2: Performing statistical analysis...")
        print("\n" + "-"*70)
        print("STEP 2: Performing Statistical Analysis")
        print("-"*70)

        try:
            analyzer = SimilarityAnalyzer("data/processed/embeddings_results.json")
            results = analyzer.run_full_analysis()

            # Save results
            output_file = Path("results/statistical_analysis.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(output_file, 'w') as f:
                json.dump(results, indent=2, fp=f)

            print(f"\n✓ Statistical analysis complete")
            print(f"✓ Results saved to {output_file}")

            # Print summary
            desc = results["descriptive_statistics"]
            t_test = results["t_test"]

            print(f"\n{'='*70}")
            print("PRELIMINARY RESULTS")
            print('='*70)
            print(f"\nCouples mean similarity: {desc['couples']['mean']:.4f}")
            print(f"Random pairs mean similarity: {desc['random_pairs']['mean']:.4f}")
            print(f"\nDifference: {desc['couples']['mean'] - desc['random_pairs']['mean']:.4f}")
            print(f"P-value: {t_test['p_value']:.4f}")
            print(f"\n{t_test['interpretation']}")
            print('='*70)

        except FileNotFoundError:
            logger.error("Embeddings file not found. Run extraction first.")
            print("\n✗ Error: Embeddings file not found")
            print("Run without --skip-extraction first")
            return 1
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            print(f"\n✗ Error: {e}")
            return 1
    else:
        logger.info("Skipping statistical analysis...")
        print("\nSkipping statistical analysis (using existing results)...")

    # Step 3: Generate visualizations
    logger.info("STEP 3: Generating visualizations...")
    print("\n" + "-"*70)
    print("STEP 3: Generating Visualizations")
    print("-"*70)

    try:
        visualizer = SimilarityVisualizer("results/statistical_analysis.json")
        visualizer.create_all_visualizations()

        print(f"\n✓ Visualizations created successfully")
        print(f"✓ Check the results/figures/ directory")

    except FileNotFoundError:
        logger.error("Analysis results not found. Run analysis first.")
        print("\n✗ Error: Analysis results not found")
        print("Run without --skip-analysis first")
        return 1
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        print(f"\n✗ Error: {e}")
        return 1

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review results in results/statistical_analysis.json")
    print("2. Check visualizations in results/figures/")
    print("3. Update blog_post.md with your findings")
    print("4. Share your analysis!")
    print("\n" + "="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

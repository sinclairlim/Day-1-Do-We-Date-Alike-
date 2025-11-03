"""
Main script to extract facial embeddings from couple images.

This script processes all couples in the dataset, detects faces,
and extracts embeddings for similarity analysis.
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.facial_analysis import FaceDetector, FaceEmbeddingExtractor
from src.data_collection import CoupleDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_all_couples(
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    detector_method: str = "opencv",
    model_name: str = "facenet"
):
    """
    Process all couples: detect faces and extract embeddings.

    Args:
        data_dir: Directory containing raw couple images
        processed_dir: Directory to save processed data
        detector_method: Face detection method
        model_name: Embedding model name
    """
    # Initialize components
    collector = CoupleDataCollector(data_dir)
    detector = FaceDetector(method=detector_method)
    extractor = FaceEmbeddingExtractor(model_name=model_name)

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    couples = collector.list_couples()
    results = []

    logger.info(f"Processing {len(couples)} couples...")

    for i, couple in enumerate(couples):
        couple_id = couple["id"]
        logger.info(f"Processing couple {i+1}/{len(couples)}: {couple_id}")

        try:
            # Extract faces
            person1_img, person2_img = collector.get_couple_images(couple_id)

            # Check if faces are already extracted (160x160) or need extraction
            import cv2
            img1 = cv2.imread(person1_img)
            img2 = cv2.imread(person2_img)

            # If images are already 160x160, they're pre-extracted faces
            if img1.shape[:2] == (160, 160) and img2.shape[:2] == (160, 160):
                logger.info(f"  Using pre-extracted faces (160x160)")
                # Extract embeddings directly from file paths for best compatibility
                emb1 = extractor.extract_from_file(person1_img)
                emb2 = extractor.extract_from_file(person2_img)
            else:
                # Need to detect and extract faces first
                logger.info(f"  Detecting and extracting faces")
                face1 = detector.extract_face(person1_img)
                face2 = detector.extract_face(person2_img)

                if face1 is None or face2 is None:
                    logger.warning(f"Could not detect faces for couple {couple_id}")
                    continue

                # Extract embeddings from face arrays
                emb1 = extractor.extract_embedding(face1)
                emb2 = extractor.extract_embedding(face2)

            # Save embeddings
            couple_proc_dir = processed_path / couple_id
            couple_proc_dir.mkdir(exist_ok=True)

            extractor.save_embedding(emb1, str(couple_proc_dir / "person1_embedding.npy"))
            extractor.save_embedding(emb2, str(couple_proc_dir / "person2_embedding.npy"))

            # Calculate similarity
            similarity = extractor.compute_similarity(emb1, emb2, metric="cosine")

            result = {
                "couple_id": couple_id,
                "similarity": float(similarity),
                "embedding1_path": str(couple_proc_dir / "person1_embedding.npy"),
                "embedding2_path": str(couple_proc_dir / "person2_embedding.npy"),
                **couple
            }

            results.append(result)
            logger.info(f"  Similarity: {similarity:.4f}")

        except Exception as e:
            logger.error(f"Error processing couple {couple_id}: {e}")
            continue

    # Save results
    results_file = processed_path / "embeddings_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    logger.info(f"\nProcessed {len(results)}/{len(couples)} couples successfully")
    logger.info(f"Results saved to {results_file}")

    return results


def main():
    """Run the embedding extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract facial embeddings from couple images"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw couple images"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save processed embeddings"
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

    results = process_all_couples(
        data_dir=args.data_dir,
        processed_dir=args.output_dir,
        detector_method=args.detector,
        model_name=args.model
    )

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Successfully processed: {len(results)} couples")
    if results:
        similarities = [r["similarity"] for r in results]
        print(f"Average similarity: {np.mean(similarities):.4f}")
        print(f"Similarity range: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example usage script demonstrating how to use the modules.

This script shows basic usage of each component without running
the full pipeline.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_collection import CoupleDataCollector
from src.facial_analysis import FaceDetector, FaceEmbeddingExtractor, SimilarityCalculator


def example_data_collection():
    """Example: Adding couples to the dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Collection")
    print("="*60)

    collector = CoupleDataCollector()

    # Option 1: Add from local files
    # collector.add_couple_from_files(
    #     "path/to/person1.jpg",
    #     "path/to/person2.jpg",
    #     metadata={"names": "John & Jane", "source": "public_dataset"}
    # )

    # Option 2: Add from URLs
    # collector.add_couple_from_urls(
    #     "https://example.com/person1.jpg",
    #     "https://example.com/person2.jpg",
    #     metadata={"celebrity": True}
    # )

    print(f"Current dataset size: {collector.get_couple_count()} couples")
    print("\nUncomment the code above to add your first couple!")


def example_face_detection():
    """Example: Detecting faces in an image."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Face Detection")
    print("="*60)

    # Initialize detector
    detector = FaceDetector(method="opencv")

    # Detect faces in an image
    image_path = "path/to/your/image.jpg"

    print(f"\nDetecting faces in: {image_path}")
    print("(Replace with actual image path to test)")

    # faces = detector.detect_faces(image_path)
    # print(f"Found {len(faces)} face(s)")

    # Extract the primary face
    # face = detector.extract_face(image_path)
    # if face is not None:
    #     print(f"Extracted face shape: {face.shape}")

    print("\nAvailable detection methods:")
    print("  - opencv (fast, good for clear images)")
    print("  - mtcnn (accurate, handles various angles)")
    print("  - dlib (robust, requires dlib installation)")


def example_embedding_extraction():
    """Example: Extracting facial embeddings."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Embedding Extraction")
    print("="*60)

    # Initialize extractor
    print("\nInitializing FaceNet model...")
    # extractor = FaceEmbeddingExtractor(model_name="facenet")

    # Extract embedding from a face image
    # embedding = extractor.extract_from_file("path/to/face.jpg")
    # if embedding is not None:
    #     print(f"Embedding shape: {embedding.shape}")
    #     print(f"Embedding dimension: {len(embedding)}")

    print("\nAvailable models:")
    print("  - facenet (128-dim, fast, recommended)")
    print("  - vggface (2622-dim, accurate)")
    print("  - arcface (512-dim, state-of-the-art)")


def example_similarity_calculation():
    """Example: Calculating facial similarity."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Similarity Calculation")
    print("="*60)

    # Example embeddings (replace with actual embeddings)
    import numpy as np

    # Simulate two face embeddings
    embedding1 = np.random.randn(128)
    embedding2 = np.random.randn(128)

    # Calculate similarity
    calc = SimilarityCalculator()

    cosine_sim = calc.cosine_similarity(embedding1, embedding2)
    euclidean_dist = calc.euclidean_distance(embedding1, embedding2)

    print(f"\nExample similarity metrics:")
    print(f"  Cosine similarity: {cosine_sim:.4f} (range: -1 to 1, higher = more similar)")
    print(f"  Euclidean distance: {euclidean_dist:.4f} (lower = more similar)")

    print("\nFor real usage:")
    print("  1. Extract embeddings for two faces")
    print("  2. Calculate similarity using cosine similarity")
    print("  3. Threshold: typically 0.6-0.7 for 'same person'")


def example_full_workflow():
    """Example: Complete workflow for one couple."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Complete Workflow")
    print("="*60)

    print("""
Complete workflow for analyzing one couple:

1. Add couple to dataset:
   collector = CoupleDataCollector()
   collector.add_couple_from_files("person1.jpg", "person2.jpg")

2. Detect and extract faces:
   detector = FaceDetector()
   face1 = detector.extract_face("person1.jpg")
   face2 = detector.extract_face("person2.jpg")

3. Extract embeddings:
   extractor = FaceEmbeddingExtractor()
   emb1 = extractor.extract_embedding(face1)
   emb2 = extractor.extract_embedding(face2)

4. Calculate similarity:
   similarity = extractor.compute_similarity(emb1, emb2)
   print(f"Similarity: {similarity:.4f}")

5. Compare with random pairs to determine if significant

See run_pipeline.py for automated processing of multiple couples!
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DO WE DATE ALIKE? - Example Usage")
    print("="*70)

    example_data_collection()
    example_face_detection()
    example_embedding_extraction()
    example_similarity_calculation()
    example_full_workflow()

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Add couple images to data/raw/")
    print("2. Run: python run_pipeline.py")
    print("3. Check results in results/")
    print("\nFor more details, see SETUP.md")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

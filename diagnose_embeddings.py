#!/usr/bin/env python3
"""
Diagnostic script to debug why embeddings are identical.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent))

from src.facial_analysis import FaceEmbeddingExtractor

def diagnose_couple(couple_id: str = "couple_0000"):
    """Diagnose a single couple's embeddings."""
    print("=" * 70)
    print(f"DIAGNOSING {couple_id}")
    print("=" * 70)

    # Load images
    person1_path = f"data/raw/{couple_id}/person1.jpg"
    person2_path = f"data/raw/{couple_id}/person2.jpg"

    img1 = cv2.imread(person1_path)
    img2 = cv2.imread(person2_path)

    print(f"\nImage 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    # Check if images are identical
    pixel_diff = np.sum(np.abs(img1.astype(float) - img2.astype(float)))
    print(f"Pixel difference: {pixel_diff}")

    if pixel_diff == 0:
        print("⚠️  WARNING: Images are IDENTICAL!")
        return

    # Check image statistics
    print(f"\nImage 1 - BGR mean: {img1.mean(axis=(0,1))}")
    print(f"Image 2 - BGR mean: {img2.mean(axis=(0,1))}")

    # Convert to RGB and check
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print(f"\nImage 1 - RGB mean: {img1_rgb.mean(axis=(0,1))}")
    print(f"Image 2 - RGB mean: {img2_rgb.mean(axis=(0,1))}")

    # Extract embeddings
    print("\n" + "=" * 70)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 70)

    extractor = FaceEmbeddingExtractor(model_name="facenet")

    emb1 = extractor.extract_from_file(person1_path)
    emb2 = extractor.extract_from_file(person2_path)

    print(f"\nEmbedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")

    print(f"\nEmbedding 1 - first 10 values:")
    print(emb1[:10])

    print(f"\nEmbedding 2 - first 10 values:")
    print(emb2[:10])

    # Check if embeddings are identical
    emb_diff = np.sum(np.abs(emb1 - emb2))
    print(f"\nEmbedding difference (L1): {emb_diff}")

    if emb_diff < 0.001:
        print("⚠️  WARNING: Embeddings are NEARLY IDENTICAL!")

    # Calculate similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\nCosine similarity: {similarity}")

    # Check norms
    print(f"\nEmbedding 1 norm: {np.linalg.norm(emb1)}")
    print(f"Embedding 2 norm: {np.linalg.norm(emb2)}")

    # Check if embeddings are normalized
    if abs(np.linalg.norm(emb1) - 1.0) < 0.01:
        print("✓ Embedding 1 is L2-normalized")
    else:
        print("✗ Embedding 1 is NOT L2-normalized")

    if abs(np.linalg.norm(emb2) - 1.0) < 0.01:
        print("✓ Embedding 2 is L2-normalized")
    else:
        print("✗ Embedding 2 is NOT L2-normalized")

    # Test with raw preprocessing
    print("\n" + "=" * 70)
    print("TESTING RAW PREPROCESSING (DEBUG)")
    print("=" * 70)

    # Manual preprocessing
    img1_rgb_norm = img1_rgb / 255.0
    img2_rgb_norm = img2_rgb / 255.0

    print(f"\nNormalized Image 1 mean: {img1_rgb_norm.mean()}")
    print(f"Normalized Image 2 mean: {img2_rgb_norm.mean()}")

    img1_batch = np.expand_dims(img1_rgb_norm, axis=0)
    img2_batch = np.expand_dims(img2_rgb_norm, axis=0)

    print(f"\nBatch 1 shape: {img1_batch.shape}")
    print(f"Batch 2 shape: {img2_batch.shape}")

    # Get embeddings directly from model
    emb1_raw = extractor.model.embeddings(img1_batch)[0]
    emb2_raw = extractor.model.embeddings(img2_batch)[0]

    print(f"\nRaw embedding 1 - first 10 values:")
    print(emb1_raw[:10])

    print(f"\nRaw embedding 2 - first 10 values:")
    print(emb2_raw[:10])

    raw_similarity = np.dot(emb1_raw, emb2_raw) / (np.linalg.norm(emb1_raw) * np.linalg.norm(emb2_raw))
    print(f"\nRaw cosine similarity: {raw_similarity}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


def check_multiple_couples(n: int = 5):
    """Check multiple couples."""
    print("\n" + "=" * 70)
    print(f"CHECKING {n} COUPLES")
    print("=" * 70)

    for i in range(n):
        couple_id = f"couple_{i:04d}"
        couple_path = Path(f"data/raw/{couple_id}")

        if not couple_path.exists():
            print(f"\n{couple_id}: Does not exist")
            continue

        person1_path = couple_path / "person1.jpg"
        person2_path = couple_path / "person2.jpg"

        if not person1_path.exists() or not person2_path.exists():
            print(f"\n{couple_id}: Missing person images")
            continue

        img1 = cv2.imread(str(person1_path))
        img2 = cv2.imread(str(person2_path))

        pixel_diff = np.sum(np.abs(img1.astype(float) - img2.astype(float)))

        # Load embeddings
        emb1 = np.load(f"data/processed/{couple_id}/person1_embedding.npy")
        emb2 = np.load(f"data/processed/{couple_id}/person2_embedding.npy")

        emb_diff = np.sum(np.abs(emb1 - emb2))
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        print(f"\n{couple_id}:")
        print(f"  Pixel diff: {pixel_diff:.0f}")
        print(f"  Embedding diff: {emb_diff:.4f}")
        print(f"  Similarity: {similarity:.6f}")

        if pixel_diff > 1000 and emb_diff < 0.1:
            print(f"  ⚠️  PROBLEM: Different images but similar embeddings!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose embedding issues")
    parser.add_argument("--couple", default="couple_0000", help="Couple ID to diagnose")
    parser.add_argument("--check-multiple", type=int, default=0, help="Check N couples")

    args = parser.parse_args()

    if args.check_multiple > 0:
        check_multiple_couples(args.check_multiple)
    else:
        diagnose_couple(args.couple)


if __name__ == "__main__":
    main()

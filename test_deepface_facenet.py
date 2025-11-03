#!/usr/bin/env python3
"""
Test DeepFace's FaceNet implementation.
"""

import numpy as np
from deepface import DeepFace

def test_deepface():
    """Test DeepFace FaceNet."""
    print("=" * 70)
    print("TESTING DEEPFACE FACENET")
    print("=" * 70)

    # Extract embeddings using DeepFace
    print("\nExtracting embedding for person 1...")
    result1 = DeepFace.represent(
        img_path="data/raw/couple_0000/person1.jpg",
        model_name="Facenet",
        enforce_detection=False
    )
    emb1 = np.array(result1[0]["embedding"])

    print("Extracting embedding for person 2...")
    result2 = DeepFace.represent(
        img_path="data/raw/couple_0000/person2.jpg",
        model_name="Facenet",
        enforce_detection=False
    )
    emb2 = np.array(result2[0]["embedding"])

    print(f"\nEmbedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")

    print(f"\nEmbedding 1 first 10:")
    print(emb1[:10])

    print(f"\nEmbedding 2 first 10:")
    print(emb2[:10])

    # Check if identical
    are_identical = np.array_equal(emb1, emb2)
    print(f"\nAre embeddings identical? {are_identical}")

    if not are_identical:
        diff = np.sum(np.abs(emb1 - emb2))
        print(f"✓ GOOD! Embeddings are different (L1 diff: {diff:.4f})")

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"Cosine similarity: {similarity:.6f}")
    else:
        print("✗ BAD! Embeddings are identical!")

    # Test another couple
    print("\n" + "=" * 70)
    print("TESTING COUPLE_0001")
    print("=" * 70)

    result3 = DeepFace.represent(
        img_path="data/raw/couple_0001/person1.jpg",
        model_name="Facenet",
        enforce_detection=False
    )
    emb3 = np.array(result3[0]["embedding"])

    print(f"\nEmbedding 3 first 10:")
    print(emb3[:10])

    print(f"\nIs emb3 same as emb1? {np.array_equal(emb3, emb1)}")

    if not np.array_equal(emb3, emb1):
        diff_3_1 = np.sum(np.abs(emb3 - emb1))
        print(f"✓ GOOD! Different couples have different embeddings (L1 diff: {diff_3_1:.4f})")

        sim_3_1 = np.dot(emb3, emb1) / (np.linalg.norm(emb3) * np.linalg.norm(emb1))
        print(f"Similarity between couple_0000/person1 and couple_0001/person1: {sim_3_1:.6f}")
    else:
        print("✗ BAD! Different couples have identical embeddings!")


if __name__ == "__main__":
    test_deepface()

#!/usr/bin/env python3
"""
Test fresh embedding extraction without any caching.
"""

import numpy as np
import cv2
from keras_facenet import FaceNet

def test_direct_facenet():
    """Test FaceNet directly without our wrapper."""
    print("=" * 70)
    print("TESTING FACENET DIRECTLY")
    print("=" * 70)

    # Load model
    model = FaceNet()
    print("✓ Model loaded")

    # Load two different images
    img1 = cv2.imread("data/raw/couple_0000/person1.jpg")
    img2 = cv2.imread("data/raw/couple_0000/person2.jpg")

    print(f"\nImage 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Normalize
    img1_norm = img1_rgb.astype('float32') / 255.0
    img2_norm = img2_rgb.astype('float32') / 255.0

    print(f"\nImage 1 normalized mean: {img1_norm.mean()}")
    print(f"Image 2 normalized mean: {img2_norm.mean()}")

    # Add batch dimension
    img1_batch = np.expand_dims(img1_norm, axis=0)
    img2_batch = np.expand_dims(img2_norm, axis=0)

    # Extract embeddings
    print("\nExtracting embedding 1...")
    emb1 = model.embeddings(img1_batch)
    print("Extracting embedding 2...")
    emb2 = model.embeddings(img2_batch)

    emb1 = emb1[0]  # Remove batch dimension
    emb2 = emb2[0]

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

    # Test couple_0001
    print("\n" + "=" * 70)
    print("TESTING COUPLE_0001")
    print("=" * 70)

    img3 = cv2.imread("data/raw/couple_0001/person1.jpg")
    img4 = cv2.imread("data/raw/couple_0001/person2.jpg")

    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img4_rgb = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

    img3_norm = img3_rgb.astype('float32') / 255.0
    img4_norm = img4_rgb.astype('float32') / 255.0

    img3_batch = np.expand_dims(img3_norm, axis=0)
    img4_batch = np.expand_dims(img4_norm, axis=0)

    emb3 = model.embeddings(img3_batch)[0]
    emb4 = model.embeddings(img4_batch)[0]

    print(f"Embedding 3 first 10:")
    print(emb3[:10])

    print(f"\nEmbedding 4 first 10:")
    print(emb4[:10])

    print(f"\nIs emb3 same as emb1? {np.array_equal(emb3, emb1)}")
    print(f"Is emb4 same as emb2? {np.array_equal(emb4, emb2)}")

    if not np.array_equal(emb3, emb1):
        print("✓ GOOD! Different couples have different embeddings")
    else:
        print("✗ BAD! Different couples have identical embeddings!")


if __name__ == "__main__":
    test_direct_facenet()

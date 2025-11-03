#!/usr/bin/env python3
"""
Test FaceNet with a random noise image to see if it always returns the same output.
"""

import numpy as np
from keras_facenet import FaceNet

def test_with_random_images():
    """Test FaceNet with random noise images."""
    print("=" * 70)
    print("TESTING WITH RANDOM NOISE IMAGES")
    print("=" * 70)

    model = FaceNet()
    print("✓ Model loaded\n")

    # Create three random 160x160x3 images
    np.random.seed(42)
    random_img1 = np.random.random((160, 160, 3)).astype('float32')

    np.random.seed(123)
    random_img2 = np.random.random((160, 160, 3)).astype('float32')

    np.random.seed(999)
    random_img3 = np.random.random((160, 160, 3)).astype('float32')

    print(f"Random image 1 mean: {random_img1.mean():.4f}")
    print(f"Random image 2 mean: {random_img2.mean():.4f}")
    print(f"Random image 3 mean: {random_img3.mean():.4f}")

    # Add batch dimension
    batch1 = np.expand_dims(random_img1, axis=0)
    batch2 = np.expand_dims(random_img2, axis=0)
    batch3 = np.expand_dims(random_img3, axis=0)

    # Extract embeddings
    emb1 = model.embeddings(batch1)[0]
    emb2 = model.embeddings(batch2)[0]
    emb3 = model.embeddings(batch3)[0]

    print(f"\nEmbedding 1 first 10: {emb1[:10]}")
    print(f"Embedding 2 first 10: {emb2[:10]}")
    print(f"Embedding 3 first 10: {emb3[:10]}")

    # Check if all identical
    all_same = np.array_equal(emb1, emb2) and np.array_equal(emb2, emb3)

    if all_same:
        print("\n✗ CRITICAL BUG: Model returns SAME output for different random inputs!")
        print("This means the keras-facenet model is broken or weights are corrupted.")
    else:
        print("\n✓ Model produces different outputs for different inputs")

        # Check similarity between random noise
        sim_1_2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_1_3 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        sim_2_3 = np.dot(emb2, emb3) / (np.linalg.norm(emb2) * np.linalg.norm(emb3))

        print(f"Similarity random 1 vs 2: {sim_1_2:.6f}")
        print(f"Similarity random 1 vs 3: {sim_1_3:.6f}")
        print(f"Similarity random 2 vs 3: {sim_2_3:.6f}")


if __name__ == "__main__":
    test_with_random_images()

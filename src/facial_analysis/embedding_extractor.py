"""
Facial embedding extraction using deep learning models.

This module extracts high-dimensional facial features that can be used
to measure similarity between faces.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceEmbeddingExtractor:
    """Extracts facial embeddings using deep learning models."""

    def __init__(self, model_name: str = "facenet"):
        """
        Initialize embedding extractor.

        Args:
            model_name: Model to use ('facenet', 'vggface', or 'arcface')
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        if self.model_name == "facenet":
            self._load_facenet()
        elif self.model_name == "vggface":
            self._load_vggface()
        elif self.model_name == "arcface":
            self._load_arcface()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_facenet(self):
        """Load FaceNet model."""
        # NOTE: keras-facenet has a critical bug where it returns identical embeddings
        # for all inputs. We use DeepFace's Facenet implementation instead.
        logger.info("Using DeepFace's Facenet implementation (keras-facenet is broken)")
        try:
            from deepface import DeepFace
            self.model = "DeepFace-Facenet"
            logger.info("Loaded Facenet model via DeepFace")
        except ImportError:
            raise ImportError("Install deepface: pip install deepface")

    def _load_vggface(self):
        """Load VGGFace model."""
        try:
            from deepface import DeepFace
            self.model = "DeepFace-VGGFace"
            logger.info("Loaded VGGFace model via DeepFace")
        except ImportError:
            raise ImportError("Install deepface: pip install deepface")

    def _load_arcface(self):
        """Load ArcFace model."""
        try:
            from deepface import DeepFace
            self.model = "DeepFace-ArcFace"
            logger.info("Loaded ArcFace model via DeepFace")
        except ImportError:
            raise ImportError("Install deepface: pip install deepface")

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a face image.

        Args:
            face_image: Preprocessed face image (typically 160x160)

        Returns:
            Embedding vector (128-dim for FaceNet, 2622-dim for VGGFace)
        """
        if self.model_name == "facenet" and hasattr(self.model, 'embeddings'):
            # Using keras-facenet
            # Convert BGR to RGB (cv2.imread loads in BGR, but FaceNet expects RGB)
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image

            face_normalized = face_rgb / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            embedding = self.model.embeddings(face_expanded)
            return embedding[0]

        elif isinstance(self.model, str) and self.model.startswith("DeepFace"):
            # Using deepface wrapper
            from deepface import DeepFace
            model_name = self.model.split("-")[1]

            # DeepFace expects image in BGR format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Assume it's already BGR from cv2
                pass

            embedding_objs = DeepFace.represent(
                img_path=face_image,
                model_name=model_name,
                enforce_detection=False
            )

            return np.array(embedding_objs[0]["embedding"])

        else:
            raise ValueError("Model not properly initialized")

    def extract_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract embedding directly from image file.

        Args:
            image_path: Path to face image

        Returns:
            Embedding vector or None if extraction fails
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None

            return self.extract_embedding(img)

        except Exception as e:
            logger.error(f"Error extracting embedding from {image_path}: {e}")
            return None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score (higher = more similar for cosine,
                            lower = more similar for euclidean)
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        elif metric == "euclidean":
            # Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save_embedding(self, embedding: np.ndarray, output_path: str):
        """Save embedding to file."""
        np.save(output_path, embedding)
        logger.info(f"Saved embedding to {output_path}")

    def load_embedding(self, embedding_path: str) -> np.ndarray:
        """Load embedding from file."""
        return np.load(embedding_path)


class SimilarityCalculator:
    """Calculate various similarity metrics between faces."""

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity (0-1, higher is more similar)."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Euclidean distance (lower is more similar)."""
        return float(np.linalg.norm(emb1 - emb2))

    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Manhattan distance (L1 norm)."""
        return float(np.sum(np.abs(emb1 - emb2)))

    @staticmethod
    def normalize_to_similarity(
        distance: float,
        metric: str = "euclidean",
        threshold: float = 10.0
    ) -> float:
        """
        Convert distance to similarity score (0-1).

        Args:
            distance: Distance value
            metric: Type of distance metric used
            threshold: Distance threshold for normalization

        Returns:
            Similarity score between 0 and 1
        """
        if metric in ["euclidean", "manhattan"]:
            # Convert distance to similarity
            similarity = 1 / (1 + distance / threshold)
            return similarity
        return distance


def main():
    """Example usage of embedding extractor."""
    print("Face Embedding Extractor")
    print("Models available: facenet, vggface, arcface")
    print("\nUsage:")
    print("  extractor = FaceEmbeddingExtractor('facenet')")
    print("  embedding = extractor.extract_from_file('face.jpg')")
    print("  similarity = extractor.compute_similarity(emb1, emb2)")


if __name__ == "__main__":
    main()

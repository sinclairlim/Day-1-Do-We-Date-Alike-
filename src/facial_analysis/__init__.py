"""Facial analysis module for face detection and embedding extraction."""

from .face_detector import FaceDetector
from .embedding_extractor import FaceEmbeddingExtractor, SimilarityCalculator

__all__ = ['FaceDetector', 'FaceEmbeddingExtractor', 'SimilarityCalculator']

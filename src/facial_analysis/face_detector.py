"""
Face detection module using MTCNN and other detection methods.

This module provides robust face detection capabilities for preprocessing
couple images before similarity analysis.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects and extracts faces from images."""

    def __init__(self, method: str = "opencv"):
        """
        Initialize face detector.

        Args:
            method: Detection method ('opencv', 'mtcnn', or 'dlib')
        """
        self.method = method
        self.detector = self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the appropriate face detector."""
        if self.method == "opencv":
            # Using Haar Cascade (lightweight, fast)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)

        elif self.method == "mtcnn":
            try:
                from mtcnn import MTCNN
                return MTCNN()
            except ImportError:
                logger.warning("MTCNN not installed. Install with: pip install mtcnn")
                raise

        elif self.method == "dlib":
            try:
                import dlib
                return dlib.get_frontal_face_detector()
            except ImportError:
                logger.warning("dlib not installed. Install with: pip install dlib")
                raise

        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        if self.method == "opencv":
            return self._detect_opencv(image_path)
        elif self.method == "mtcnn":
            return self._detect_mtcnn(image_path)
        elif self.method == "dlib":
            return self._detect_dlib(image_path)

    def _detect_opencv(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascade."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _detect_mtcnn(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img_rgb)

        faces = []
        for result in results:
            x, y, w, h = result['box']
            faces.append((x, y, w, h))

        return faces

    def _detect_dlib(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        return [(face.left(), face.top(),
                 face.right() - face.left(),
                 face.bottom() - face.top()) for face in faces]

    def extract_face(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (160, 160),
        padding: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Extract and align the largest face from an image.

        Args:
            image_path: Path to the image
            target_size: Size to resize the face to
            padding: Padding around face (as fraction of face size)

        Returns:
            Extracted face as numpy array, or None if no face found
        """
        faces = self.detect_faces(image_path)

        if not faces:
            logger.warning(f"No faces detected in {image_path}")
            return None

        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Ensure we don't go out of bounds
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)

        # Extract face
        face = img[y1:y2, x1:x2]

        # Resize to target size
        face = cv2.resize(face, target_size)

        return face

    def extract_and_save_face(
        self,
        image_path: str,
        output_path: str,
        target_size: Tuple[int, int] = (160, 160)
    ) -> bool:
        """
        Extract face and save to file.

        Args:
            image_path: Input image path
            output_path: Output path for extracted face
            target_size: Size to resize face to

        Returns:
            True if successful, False otherwise
        """
        face = self.extract_face(image_path, target_size)

        if face is None:
            return False

        cv2.imwrite(output_path, face)
        logger.info(f"Saved extracted face to {output_path}")
        return True


def main():
    """Example usage of face detector."""
    detector = FaceDetector(method="opencv")

    print("Face Detector initialized")
    print("Methods available: opencv, mtcnn, dlib")
    print("\nUsage:")
    print("  faces = detector.detect_faces('path/to/image.jpg')")
    print("  face = detector.extract_face('path/to/image.jpg')")


if __name__ == "__main__":
    main()

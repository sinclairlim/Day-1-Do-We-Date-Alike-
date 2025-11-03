"""
Data collection module for gathering couple images.

This module provides utilities to collect and organize couple images
from various sources for facial similarity analysis.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoupleDataCollector:
    """Collects and manages couple image data."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data collector.

        Args:
            data_dir: Directory to store collected images
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"couples": []}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)

    def add_couple_from_single_photo(
        self,
        couple_photo_path: str,
        couple_id: str = None,
        metadata: Dict = None,
        detector_method: str = "opencv"
    ) -> str:
        """
        Add a couple from a SINGLE photo containing both people.

        This method will automatically detect both faces and extract them.

        Args:
            couple_photo_path: Path to photo containing both people
            couple_id: Optional custom ID for the couple
            metadata: Optional additional metadata (names, source, etc.)
            detector_method: Face detection method ('opencv', 'mtcnn', 'dlib')

        Returns:
            couple_id: ID assigned to this couple

        Raises:
            ValueError: If exactly 2 faces are not detected
        """
        # Import here to avoid circular dependency
        import sys
        import cv2
        sys.path.append(str(Path(__file__).parent.parent))
        from facial_analysis.face_detector import FaceDetector

        if couple_id is None:
            couple_id = f"couple_{len(self.metadata['couples']):04d}"

        couple_dir = self.data_dir / couple_id
        couple_dir.mkdir(exist_ok=True)

        # Detect faces
        detector = FaceDetector(method=detector_method)
        faces = detector.detect_faces(couple_photo_path)

        if len(faces) == 0:
            raise ValueError(f"No faces detected in {couple_photo_path}")
        elif len(faces) == 1:
            raise ValueError(f"Only 1 face detected in {couple_photo_path}. Need exactly 2 faces.")
        elif len(faces) > 2:
            logger.warning(f"Found {len(faces)} faces in {couple_photo_path}. Using the 2 largest.")
            # Sort by face size (area) and take the 2 largest
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]

        # Extract both faces
        img = cv2.imread(couple_photo_path)

        for idx, (x, y, w, h) in enumerate(faces, 1):
            # Add padding
            padding = 0.2
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            height, width = img.shape[:2]

            # Ensure we don't go out of bounds
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + w + pad_w)
            y2 = min(height, y + h + pad_h)

            # Extract face
            face = img[y1:y2, x1:x2]

            # Resize to standard size
            face_resized = cv2.resize(face, (160, 160))

            # Save
            output_path = couple_dir / f"person{idx}.jpg"
            cv2.imwrite(str(output_path), face_resized)

        # Store metadata
        couple_metadata = {
            "id": couple_id,
            "person1_image": str(couple_dir / "person1.jpg"),
            "person2_image": str(couple_dir / "person2.jpg"),
            "source_image": couple_photo_path,
            "extraction_method": "automatic_dual_face",
            **(metadata or {})
        }

        self.metadata["couples"].append(couple_metadata)
        self._save_metadata()

        logger.info(f"Added couple {couple_id} (extracted 2 faces from single photo)")
        return couple_id

    def add_couple_from_files(
        self,
        person1_path: str,
        person2_path: str,
        couple_id: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Add a couple from local image files.

        Args:
            person1_path: Path to first person's image
            person2_path: Path to second person's image
            couple_id: Optional custom ID for the couple
            metadata: Optional additional metadata (names, source, etc.)

        Returns:
            couple_id: ID assigned to this couple
        """
        if couple_id is None:
            couple_id = f"couple_{len(self.metadata['couples']):04d}"

        couple_dir = self.data_dir / couple_id
        couple_dir.mkdir(exist_ok=True)

        # Copy images
        img1 = Image.open(person1_path)
        img2 = Image.open(person2_path)

        img1.save(couple_dir / "person1.jpg")
        img2.save(couple_dir / "person2.jpg")

        # Store metadata
        couple_metadata = {
            "id": couple_id,
            "person1_image": str(couple_dir / "person1.jpg"),
            "person2_image": str(couple_dir / "person2.jpg"),
            **(metadata or {})
        }

        self.metadata["couples"].append(couple_metadata)
        self._save_metadata()

        logger.info(f"Added couple {couple_id}")
        return couple_id

    def add_couple_from_urls(
        self,
        person1_url: str,
        person2_url: str,
        couple_id: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Add a couple from image URLs.

        Args:
            person1_url: URL to first person's image
            person2_url: URL to second person's image
            couple_id: Optional custom ID for the couple
            metadata: Optional additional metadata

        Returns:
            couple_id: ID assigned to this couple
        """
        if couple_id is None:
            couple_id = f"couple_{len(self.metadata['couples']):04d}"

        couple_dir = self.data_dir / couple_id
        couple_dir.mkdir(exist_ok=True)

        # Download images
        for idx, url in enumerate([person1_url, person2_url], 1):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                img = Image.open(requests.get(url, stream=True).raw)
                img.save(couple_dir / f"person{idx}.jpg")
            except Exception as e:
                logger.error(f"Error downloading image from {url}: {e}")
                raise

        # Store metadata
        couple_metadata = {
            "id": couple_id,
            "person1_image": str(couple_dir / "person1.jpg"),
            "person2_image": str(couple_dir / "person2.jpg"),
            "person1_url": person1_url,
            "person2_url": person2_url,
            **(metadata or {})
        }

        self.metadata["couples"].append(couple_metadata)
        self._save_metadata()

        logger.info(f"Added couple {couple_id}")
        return couple_id

    def get_couple_count(self) -> int:
        """Get total number of couples in dataset."""
        return len(self.metadata["couples"])

    def get_couple_images(self, couple_id: str) -> Tuple[str, str]:
        """
        Get image paths for a specific couple.

        Args:
            couple_id: ID of the couple

        Returns:
            Tuple of (person1_path, person2_path)
        """
        for couple in self.metadata["couples"]:
            if couple["id"] == couple_id:
                return couple["person1_image"], couple["person2_image"]
        raise ValueError(f"Couple {couple_id} not found")

    def list_couples(self) -> List[Dict]:
        """Get list of all couples with metadata."""
        return self.metadata["couples"]


def main():
    """Example usage of the data collector."""
    collector = CoupleDataCollector()

    print(f"Current dataset contains {collector.get_couple_count()} couples")
    print("\nTo add couples, use:")
    print("  collector.add_couple_from_files(path1, path2)")
    print("  collector.add_couple_from_urls(url1, url2)")
    print("\nFor bulk collection, consider:")
    print("  - Celebrity couple datasets")
    print("  - Public couple photo repositories")
    print("  - Dating app research datasets (with permission)")


if __name__ == "__main__":
    main()

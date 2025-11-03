#!/usr/bin/env python3
"""
Easy script to add couple photos to the dataset.

This script automatically extracts 2 faces from each couple photo
and organizes them for analysis.

Usage:
    # Add a single couple photo
    python add_couple_photos.py path/to/couple_photo.jpg

    # Add all photos from a directory
    python add_couple_photos.py path/to/couples_folder/

    # Use a different face detector
    python add_couple_photos.py path/to/photo.jpg --detector mtcnn
"""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_collection import CoupleDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def add_single_photo(photo_path: str, detector: str = "opencv"):
    """Add a single couple photo."""
    collector = CoupleDataCollector()

    try:
        couple_id = collector.add_couple_from_single_photo(
            photo_path,
            detector_method=detector
        )
        print(f"✓ Successfully added {couple_id}")
        return True
    except ValueError as e:
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def add_directory(dir_path: str, detector: str = "opencv"):
    """Add all couple photos from a directory."""
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        print(f"✗ Error: {dir_path} is not a directory")
        return

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))

    if not image_files:
        print(f"✗ No image files found in {dir_path}")
        return

    print(f"\nFound {len(image_files)} image(s) in {dir_path}")
    print("="*60)

    collector = CoupleDataCollector()
    successes = 0
    failures = 0

    for img_file in sorted(image_files):
        print(f"\nProcessing: {img_file.name}")

        try:
            couple_id = collector.add_couple_from_single_photo(
                str(img_file),
                detector_method=detector
            )
            print(f"  ✓ Added as {couple_id}")
            successes += 1

        except ValueError as e:
            print(f"  ✗ Skipped: {e}")
            failures += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failures += 1

    print("\n" + "="*60)
    print(f"Summary: {successes} successful, {failures} failed")
    print(f"Total couples in dataset: {collector.get_couple_count()}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Add couple photos to the dataset (auto-extracts 2 faces)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a single photo
  python add_couple_photos.py my_couple.jpg

  # Add all photos from a folder
  python add_couple_photos.py couples_folder/

  # Use MTCNN detector (more accurate)
  python add_couple_photos.py photo.jpg --detector mtcnn

Tips:
  - Photos should contain exactly 2 faces
  - If more than 2 faces detected, the 2 largest will be used
  - Faces should be clearly visible and front-facing
  - Good lighting helps detection accuracy
        """
    )

    parser.add_argument(
        'path',
        help='Path to a single photo or directory of photos'
    )

    parser.add_argument(
        '--detector',
        choices=['opencv', 'mtcnn', 'dlib'],
        default='opencv',
        help='Face detection method (default: opencv)'
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"✗ Error: {path} does not exist")
        return 1

    print("="*60)
    print("COUPLE PHOTO PROCESSOR")
    print("="*60)
    print(f"Detector: {args.detector}")
    print(f"Input: {path}")
    print("="*60)

    if path.is_file():
        # Single photo
        success = add_single_photo(str(path), args.detector)
        return 0 if success else 1

    elif path.is_dir():
        # Directory of photos
        add_directory(str(path), args.detector)
        return 0

    else:
        print(f"✗ Error: {path} is not a file or directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())

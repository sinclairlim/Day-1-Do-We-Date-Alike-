#!/usr/bin/env python3
"""
Debug tool to visualize face detection and extraction.
Shows what faces are being detected and how they're being cropped.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.facial_analysis import FaceDetector


def visualize_detection(image_path: str, detector_method: str = "opencv"):
    """Visualize face detection on an image."""
    print(f"Analyzing: {image_path}")
    print(f"Detector: {detector_method}")
    print("="*60)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    print(f"Image size: {img.shape}")

    # Detect faces
    detector = FaceDetector(method=detector_method)
    faces = detector.detect_faces(image_path)

    print(f"Found {len(faces)} face(s)")

    # Draw rectangles on image
    img_viz = img.copy()

    for i, (x, y, w, h) in enumerate(faces, 1):
        print(f"\nFace {i}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Area: {w*h}")

        # Draw detected face box (RED)
        cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_viz, f"Face {i}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate padded region (GREEN)
        padding = 0.2
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        height, width = img.shape[:2]
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)

        print(f"  With padding: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Padded size: {x2-x1}x{y2-y1}")

        # Draw padded box (GREEN)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract and save face
        face_crop = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (160, 160))

        output_path = f"debug_face_{i}.jpg"
        cv2.imwrite(output_path, face_resized)
        print(f"  Saved extracted face to: {output_path}")

    # Save visualization
    viz_path = "debug_visualization.jpg"
    cv2.imwrite(viz_path, img_viz)
    print(f"\nSaved visualization to: {viz_path}")
    print(f"  RED boxes = detected face regions")
    print(f"  GREEN boxes = with 20% padding (what gets extracted)")

    return faces


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug face detection and extraction"
    )
    parser.add_argument("image", help="Path to couple photo")
    parser.add_argument(
        "--detector",
        default="opencv",
        choices=["opencv", "mtcnn", "dlib"],
        help="Face detection method"
    )

    args = parser.parse_args()

    faces = visualize_detection(args.image, args.detector)

    print("\n" + "="*60)
    print("Check the output files:")
    print("  - debug_visualization.jpg (shows detected regions)")
    print("  - debug_face_1.jpg, debug_face_2.jpg (extracted faces)")
    print("="*60)


if __name__ == "__main__":
    main()

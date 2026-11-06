"""
Q1(a) 2D Transformations
Applies translation, rotation, scaling, and shearing on an image
using OpenCV affine transformation matrices and saves the outputs.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_translation(image, tx=70, ty=50):
    """Translate image by (tx, ty) pixels."""
    h, w = image.shape[:2]
    # 2x3 translation matrix
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    translated = cv2.warpAffine(image, M, (w, h), borderValue=(240, 240, 240))
    return translated, M


def apply_rotation(image, angle_deg=35, scale=1.0):
    """Rotate image around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # OpenCV builds the 2x3 rotation matrix for the specified center
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(240, 240, 240))
    return rotated, M


def apply_scaling(image, sx=0.7, sy=0.7):
    """Scale image using an affine scaling matrix."""
    h, w = image.shape[:2]
    # Scaling matrix centered or relative to origin
    M = np.float32([
        [sx, 0,  (1 - sx) * (w / 2)],
        [0,  sy, (1 - sy) * (h / 2)]
    ])
    scaled = cv2.warpAffine(image, M, (w, h), borderValue=(240, 240, 240))
    return scaled, M


def apply_shearing(image, shx=0.3, shy=0.1):
    """Apply horizontal and vertical shear."""
    h, w = image.shape[:2]
    # Shearing matrix: x' = x + shx*y, y' = y + shy*x
    # We add an offset to keep the sheared object within the frame
    M = np.float32([
        [1,   shx, -shx * (h * 0.25)],
        [shy, 1,   0]
    ])
    sheared = cv2.warpAffine(image, M, (w, h), borderValue=(240, 240, 240))
    return sheared, M


def main():
    parser = argparse.ArgumentParser(description="Q1(a) - 2D Image Transformations")
    parser.add_argument("--image", type=str, default="input/sample_image.png",
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output/q1a",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image from '{args.image}'. Please verify the path.")

    print(f"[Q1a] Loaded input image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    # 1. Translation
    translated, M_trans = apply_translation(img, tx=80, ty=60)
    print("\n--- Translation Matrix [2x3] ---")
    print(M_trans)

    # 2. Rotation
    rotated, M_rot = apply_rotation(img, angle_deg=35)
    print("\n--- Rotation Matrix [2x3] (35 deg) ---")
    print(M_rot)

    # 3. Scaling
    scaled, M_scale = apply_scaling(img, sx=0.7, sy=0.7)
    print("\n--- Scaling Matrix [2x3] (sx=0.7, sy=0.7) ---")
    print(M_scale)

    # 4. Shearing
    sheared, M_shear = apply_shearing(img, shx=0.35, shy=0.1)
    print("\n--- Shearing Matrix [2x3] (shx=0.35, shy=0.1) ---")
    print(M_shear)

    # Save individual results
    cv2.imwrite(os.path.join(args.output_dir, "0_original.png"), img)
    cv2.imwrite(os.path.join(args.output_dir, "1_translation.png"), translated)
    cv2.imwrite(os.path.join(args.output_dir, "2_rotation.png"), rotated)
    cv2.imwrite(os.path.join(args.output_dir, "3_scaling.png"), scaled)
    cv2.imwrite(os.path.join(args.output_dir, "4_shearing.png"), sheared)

    # Save side-by-side comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    items = [
        ("Original", img),
        ("1. Translation (+80px, +60px)", translated),
        ("2. Rotation (35 deg)", rotated),
        ("3. Scaling (0.7x)", scaled),
        ("4. Shearing (shx=0.35, shy=0.1)", sheared),
    ]

    for i, (title, pic) in enumerate(items):
        pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        axes[i].imshow(pic_rgb)
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].axis("on")
        axes[i].grid(True, linestyle=":", alpha=0.5)

    # Hide unused subplot
    axes[5].axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, "comparison_grid.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[Q1a] Successfully saved all outputs to '{args.output_dir}'")
    print(f"  - Individual images: 0_original.png ... 4_shearing.png")
    print(f"  - Comparison grid: {comparison_path}")


if __name__ == "__main__":
    main()


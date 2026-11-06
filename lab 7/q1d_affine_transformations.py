"""
Q1(d) Affine Transformations
Applies an affine transformation to an image and examines whether
straight lines remain straight after the transformation.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_affine_transform(image, src_tri, dst_tri):
    """
    Computes 2x3 affine matrix from 3 pairs of corresponding points
    and warps the image.
    """
    h, w = image.shape[:2]
    # cv2.getAffineTransform computes the 2x3 affine matrix from 3 point correspondences
    M = cv2.getAffineTransform(src_tri, dst_tri)
    transformed = cv2.warpAffine(image, M, (w, h), borderValue=(245, 245, 245))
    return transformed, M


def draw_grid_lines_and_markers(img, pts):
    """Draw straight reference lines and 3 affine control points."""
    annotated = img.copy()
    # Draw reference straight lines across the image
    h, w = img.shape[:2]
    cv2.line(annotated, (0, 0), (w, h), (0, 0, 255), 2)          # Red diagonal
    cv2.line(annotated, (0, h), (w, 0), (0, 180, 0), 2)        # Green diagonal
    cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)  # Blue vertical

    # Mark the 3 control points
    colors = [(255, 50, 50), (50, 200, 50), (50, 50, 255)]
    labels = ["P1", "P2", "P3"]
    for i, pt in enumerate(pts):
        p = tuple(np.int32(pt))
        cv2.circle(annotated, p, 7, colors[i], -1)
        cv2.circle(annotated, p, 9, (0, 0, 0), 2)
        cv2.putText(annotated, labels[i], (p[0] + 10, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Q1(d) - Affine Transformation")
    parser.add_argument("--image", type=str, default="input/sample_image.png",
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output/q1d",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: '{args.image}'")

    h, w = img.shape[:2]

    # Define 3 non-collinear source points in original image
    src_tri = np.float32([
        [50, 50],          # Top-left
        [w - 50, 50],      # Top-right
        [100, h - 50]      # Bottom-left (skewed)
    ])

    # Define their new positions in the transformed image (rotation + scaling + shear)
    dst_tri = np.float32([
        [80, 100],         # Shifted down-right
        [w - 30, 40],      # Shifted up
        [150, h - 30]      # Sheared further right
    ])

    transformed, M = apply_affine_transform(img, src_tri, dst_tri)

    print("[Q1d] Applied 2x3 Affine Transformation Matrix:")
    print(M)

    # Prepare images with visual straight reference lines to confirm linearity
    orig_annotated = draw_grid_lines_and_markers(img, src_tri)
    trans_annotated = draw_grid_lines_and_markers(transformed, dst_tri)

    # Save outputs
    cv2.imwrite(os.path.join(args.output_dir, "original.png"), img)
    cv2.imwrite(os.path.join(args.output_dir, "original_with_reference_lines.png"), orig_annotated)
    cv2.imwrite(os.path.join(args.output_dir, "affine_transformed.png"), transformed)
    cv2.imwrite(os.path.join(args.output_dir, "affine_transformed_with_lines.png"), trans_annotated)

    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(orig_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image (with Straight Reference Lines)", fontsize=12, fontweight="bold")
    axes[0].axis("on")

    axes[1].imshow(cv2.cvtColor(trans_annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Affine Transformed Image (Lines Remain Straight)", fontsize=12, fontweight="bold")
    axes[1].axis("on")

    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, "affine_comparison.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Observation on straight lines
    observation_text = """LAB 07 - QUESTION 1(d): BEHAVIOUR OF STRAIGHT LINES UNDER AFFINE TRANSFORMATION

KEY OBSERVATION:
================
DO STRAIGHT LINES REMAIN STRAIGHT AFTER AN AFFINE TRANSFORMATION?
--> YES. Straight lines STRICTLY remain straight lines after an affine transformation.

MATHEMATICAL EXPLANATION:
-------------------------
1. PRESERVATION OF COLLINEARITY:
   An affine transformation maps any 2D point x = [x, y]^T to x' via:
       x' = A * x + t
   where A is a non-singular 2x2 matrix and t is a 2D translation vector.

   A straight line in parametric form is defined by:
       L(lambda) = p0 + lambda * v
   where p0 is a base point, v is the direction vector, and lambda is a real scalar.

   Applying the affine transformation:
       L'(lambda) = A * (p0 + lambda * v) + t
                  = (A * p0 + t) + lambda * (A * v)
                  = p0' + lambda * v'

   Because L'(lambda) is a linear function of lambda with base point p0' = (A*p0 + t)
   and constant direction vector v' = (A * v), the transformed set of points forms
   another PERFECT STRAIGHT LINE.

2. PRESERVATION OF PARALLELISM:
   Two parallel lines share the same direction vector v:
       L1: p1 + lambda * v
       L2: p2 + lambda * v
   After affine transformation:
       L1': (A * p1 + t) + lambda * (A * v)
       L2': (A * p2 + t) + lambda * (A * v)
   Both transformed lines share the same transformed direction vector v' = A * v.
   Therefore, lines that are parallel before transformation REMAIN PARALLEL after transformation.

3. RATIOS OF DISTANCES ALONG A LINE:
   The ratio of lengths of any two segments on a single straight line is preserved:
       ||p_b - p_a|| / ||p_c - p_a|| = ||p_b' - p_a'|| / ||p_c' - p_a'||

4. WHAT CHANGES:
   While collinearity and parallelism are preserved:
   - Angles between non-parallel intersecting lines can change (orthogonality is not preserved).
   - Absolute lengths, areas, and aspect ratios can change due to scaling and shearing.
"""

    print("\n" + "="*70)
    print(observation_text)
    print("="*70)

    obs_path = os.path.join(args.output_dir, "straight_lines_observation.txt")
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write(observation_text)

    print(f"\n[Q1d] Saved observation to '{obs_path}'")
    print(f"[Q1d] Saved comparison figure to '{comparison_path}'")


if __name__ == "__main__":
    main()


"""
Q1(e) Projective Transformation (Homography)
Corrects the perspective of a tilted planar surface (e.g. slanted book or document)
using a 3x3 homography matrix to unwarp it into a flat, front-facing rectangular view.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def order_quad_points(pts):
    """
    Orders 4 coordinates consistently:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of (x + y): top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference (y - x): top-right has smallest diff, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def interactive_select_corners(image):
    """Interactive mouse-click callback to select 4 corner points."""
    clone = image.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(clone, f"P{len(points)}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2)
            if len(points) == 4:
                cv2.line(clone, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)
            cv2.imshow("Select 4 Corners (TL, TR, BR, BL) - Press any key when done", clone)

    window_name = "Select 4 Corners (TL, TR, BR, BL) - Press any key when done"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError(f"Expected 4 clicked points, but got {len(points)}.")
    return np.array(points, dtype=np.float32)


def compute_destination_dimensions(src_pts):
    """Calculates width and height of destination rectangle from corner distances."""
    tl, tr, br, bl = src_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    return max_w, max_h


def compute_homography_and_warp(image, src_points, dst_size=None):
    """
    Computes 3x3 homography matrix mapping tilted src_points to rectangular destination
    and applies warpPerspective.
    """
    src_rect = order_quad_points(src_points)

    if dst_size is None:
        dw, dh = compute_destination_dimensions(src_rect)
    else:
        dw, dh = dst_size

    dst_rect = np.float32([
        [0, 0],
        [dw - 1, 0],
        [dw - 1, dh - 1],
        [0, dh - 1]
    ])

    # Compute 3x3 Homography Matrix (8 DOF)
    H, _ = cv2.findHomography(src_rect, dst_rect)
    unwarped = cv2.warpPerspective(image, H, (dw, dh), flags=cv2.INTER_CUBIC)

    return unwarped, H, src_rect, dst_rect


def main():
    parser = argparse.ArgumentParser(description="Q1(e) - Projective Transformation (Homography)")
    parser.add_argument("--image", type=str, default="input/tilted_book.png",
                        help="Path to tilted image")
    parser.add_argument("--interactive", action="store_true",
                        help="Select corners interactively via mouse clicks")
    parser.add_argument("--points", type=str, default=None,
                        help="Comma-separated coordinates: x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--output_dir", type=str, default="output/q1e",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: '{args.image}'")

    print(f"[Q1e] Loaded image: '{args.image}' ({img.shape[1]}x{img.shape[0]})")

    # Corner identification
    if args.interactive:
        print("[Q1e] Opening interactive corner selection window...")
        src_points = interactive_select_corners(img)
    elif args.points:
        raw_vals = [float(v.strip()) for v in args.points.split(",")]
        if len(raw_vals) != 8:
            raise ValueError("Expected 8 numbers for 4 (x, y) coordinates.")
        src_points = np.array(raw_vals, dtype=np.float32).reshape((4, 2))
    else:
        # Default coordinates matching the synthesized tilted book
        src_points = np.float32([
            [220, 120],  # Top-Left
            [680, 160],  # Top-Right
            [780, 560],  # Bottom-Right
            [120, 500]   # Bottom-Left
        ])
        print("[Q1e] Using default corner coordinates for synthetic tilted surface.")

    unwarped, H, src_ordered, dst_ordered = compute_homography_and_warp(img, src_points)

    print("\n--- Estimated 3x3 Homography Matrix H ---")
    print(np.round(H, 6))
    print(f"\nUnwarped destination dimensions: {unwarped.shape[1]} x {unwarped.shape[0]} px")

    # Annotate original image with selected corners
    annotated = img.copy()
    cv2.polylines(annotated, [np.int32(src_ordered)], isClosed=True, color=(0, 0, 255), thickness=3)
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    for i, pt in enumerate(src_ordered):
        p = tuple(np.int32(pt))
        cv2.circle(annotated, p, 8, (0, 255, 0), -1)
        cv2.putText(annotated, f" {corner_labels[i]}", (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Save images
    cv2.imwrite(os.path.join(args.output_dir, "original_tilted.png"), img)
    cv2.imwrite(os.path.join(args.output_dir, "original_with_corners.png"), annotated)
    cv2.imwrite(os.path.join(args.output_dir, "corrected_homography.png"), unwarped)

    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input: Tilted Perspective with Detected Corners", fontsize=12, fontweight="bold")
    axes[0].axis("on")

    axes[1].imshow(cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Output: Corrected Front-Facing View (Homography)", fontsize=12, fontweight="bold")
    axes[1].axis("on")

    plt.tight_layout()
    comp_path = os.path.join(args.output_dir, "homography_comparison.png")
    plt.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[Q1e] Saved outputs to '{args.output_dir}':")
    print(f"  - original_with_corners.png")
    print(f"  - corrected_homography.png")
    print(f"  - homography_comparison.png")


if __name__ == "__main__":
    main()


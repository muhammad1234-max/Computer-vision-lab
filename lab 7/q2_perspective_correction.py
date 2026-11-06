"""
Q2: Perspective Correction (Unwarping)
Implements perspective correction by mapping a perspective-distorted surface
(tilted view as SOURCE) back into a front-facing perfect rectangle (as DESTINATION).
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def order_corner_points(pts):
    """
    Orders 4 points into consistent format:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: Top-Left has smallest sum, Bottom-Right has largest
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    # Difference (y - x): Top-Right has smallest diff, Bottom-Left has largest
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def apply_perspective_transform(image, src_points, dst_points=None, dst_size=None):
    """
    Performs perspective unwarping (inverse perspective correction).

    CRITICAL DIRECTION SPECIFICATION (from Lab 07 Tasks PDF):
    - SOURCE (src_points): The 4 corners on the tilted, perspective-distorted surface.
    - DESTINATION (dst_points): The 4 corners of the perfect, head-on rectangle.
    
    This function takes the tilted view as SOURCE and maps it directly to a
    perfect rectangle as the DESTINATION using cv2.getPerspectiveTransform().
    """
    src = order_corner_points(src_points)

    if dst_points is None:
        # Determine optimal rectangular destination size from source quad dimensions
        tl, tr, br, bl = src

        # Width is max of top and bottom edges
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        dst_w = int(max(width_top, width_bottom))

        # Height is max of left and right edges
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        dst_h = int(max(height_left, height_right))

        if dst_size is not None:
            dst_w, dst_h = dst_size

        # Define destination as a perfect rectangle
        dst = np.float32([
            [0, 0],              # Top-left
            [dst_w - 1, 0],      # Top-right
            [dst_w - 1, dst_h - 1],  # Bottom-right
            [0, dst_h - 1]       # Bottom-left
        ])
    else:
        dst = order_corner_points(dst_points)
        dst_w = int(np.max(dst[:, 0])) + 1
        dst_h = int(np.max(dst[:, 1])) + 1

    # Transformation matrix: maps tilted SOURCE -> flat rectangular DESTINATION
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp perspective to produce front-facing unwarped view
    corrected = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)

    return corrected, M, src, dst


def interactive_pick_corners(image):
    """Interactive GUI window to select 4 corner points with mouse clicks."""
    clone = image.copy()
    clicked_pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
            clicked_pts.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(clone, f"{len(clicked_pts)}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(clicked_pts) > 1:
                cv2.line(clone, tuple(clicked_pts[-2]), tuple(clicked_pts[-1]), (255, 100, 0), 2)
            if len(clicked_pts) == 4:
                cv2.line(clone, tuple(clicked_pts[3]), tuple(clicked_pts[0]), (255, 100, 0), 2)
            cv2.imshow("Click 4 Corners (TL -> TR -> BR -> BL) | Press any key when done", clone)

    win = "Click 4 Corners (TL -> TR -> BR -> BL) | Press any key when done"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_pts) != 4:
        raise ValueError(f"Expected 4 corners, but received {len(clicked_pts)}.")
    return np.array(clicked_pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Q2 - Perspective Correction (Unwarping)")
    parser.add_argument("--image", type=str, default="input/document_tilted.png",
                        help="Path to perspective-distorted image")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive mouse click corner picking")
    parser.add_argument("--output_dir", type=str, default="output/q2",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: '{args.image}'")

    print(f"[Q2] Loaded image: '{args.image}' ({img.shape[1]}x{img.shape[0]})")

    if args.interactive:
        print("[Q2] Click the 4 corners of the document in the OpenCV window...")
        src_points = interactive_pick_corners(img)
    else:
        # Default corners matching the generated tilted document
        src_points = np.float32([
            [220, 120],  # Top-Left
            [680, 160],  # Top-Right
            [780, 560],  # Bottom-Right
            [120, 500]   # Bottom-Left
        ])
        print("[Q2] Using preset 4-corner coordinates for the tilted document.")

    # Apply perspective correction with tilted view as SOURCE and rectangle as DESTINATION
    corrected, M, src_ordered, dst_ordered = apply_perspective_transform(img, src_points)

    print("\n--- Perspective Transform Matrix M (Tilted Source -> Flat Destination) ---")
    print(np.round(M, 6))
    print(f"\nCorrected image size: {corrected.shape[1]} x {corrected.shape[0]} px")

    # Annotate source image with corner polygon
    annotated = img.copy()
    cv2.polylines(annotated, [np.int32(src_ordered)], isClosed=True, color=(0, 255, 255), thickness=3)
    labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
    for i, pt in enumerate(src_ordered):
        p = tuple(np.int32(pt))
        cv2.circle(annotated, p, 8, (0, 0, 255), -1)
        cv2.putText(annotated, labels[i], (p[0] + 10, p[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save outputs
    cv2.imwrite(os.path.join(args.output_dir, "input_distorted_document.png"), img)
    cv2.imwrite(os.path.join(args.output_dir, "document_with_detected_corners.png"), annotated)
    cv2.imwrite(os.path.join(args.output_dir, "perspective_corrected_document.png"), corrected)

    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Source: Perspective-Distorted Document (Tilted View)", fontsize=12, fontweight="bold")
    axes[0].axis("on")

    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Destination: Corrected Head-On Rectangular View", fontsize=12, fontweight="bold")
    axes[1].axis("on")

    plt.tight_layout()
    side_path = os.path.join(args.output_dir, "side_by_side_correction.png")
    plt.savefig(side_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Save written explanation of the inverse mapping logic
    note = f"""LAB 07 - QUESTION 2: PERSPECTIVE CORRECTION (UNWARPING)
==============================================================
SPECIFICATION VERIFICATION:
"Modify the apply_perspective_transform function's inputs to perform an inverse transformation:
 you'll take the tilted view as the source and map it back to a perfect rectangle as the destination."

MAPPING VERIFICATION:
- SOURCE POINTS (Tilted Perspective View):
  Top-Left:     ({src_ordered[0][0]:.1f}, {src_ordered[0][1]:.1f})
  Top-Right:    ({src_ordered[1][0]:.1f}, {src_ordered[1][1]:.1f})
  Bottom-Right: ({src_ordered[2][0]:.1f}, {src_ordered[2][1]:.1f})
  Bottom-Left:  ({src_ordered[3][0]:.1f}, {src_ordered[3][1]:.1f})

- DESTINATION POINTS (Unwarped Perfect Rectangle):
  Top-Left:     (0, 0)
  Top-Right:    ({dst_ordered[1][0]:.1f}, 0)
  Bottom-Right: ({dst_ordered[2][0]:.1f}, {dst_ordered[2][1]:.1f})
  Bottom-Left:  (0, {dst_ordered[3][1]:.1f})

- COMPUTED PERSPECTIVE MATRIX M (3x3):
{M}

RESULTS:
The slanted perspective has been rectified head-on with restored parallel boundaries
and readable text alignment.
"""
    note_path = os.path.join(args.output_dir, "mapping_explanation.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    print(f"\n[Q2] All outputs saved successfully to '{args.output_dir}':")
    print("  - input_distorted_document.png")
    print("  - document_with_detected_corners.png")
    print("  - perspective_corrected_document.png")
    print("  - side_by_side_correction.png")
    print("  - mapping_explanation.txt")


if __name__ == "__main__":
    main()


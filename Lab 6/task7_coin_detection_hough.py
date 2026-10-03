"""
Task 7: Coins Detection and Counting Using Hough Circle Transformation
=======================================================================
Goal: Detect and count coins in images using Hough Circle Transform.
      Handle coins of various sizes and positions automatically.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── Synthetic coin image ──────────────────────────────────────────────────────

def create_coin_image(n_coins=15, img_size=(480, 640), seed=55):
    """
    Generate a synthetic image containing circular coins of varying sizes.

    Returns:
        img     : BGR image with coins.
        gt_info : Ground-truth list of (cx, cy, radius, colour_name).
    """
    h, w   = img_size
    rng    = np.random.default_rng(seed)
    canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)   # dark surface

    # Slightly textured background
    noise = rng.integers(0, 20, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)

    coin_colours = [
        ((180, 170, 100), "Gold"),
        ((160, 160, 160), "Silver"),
        ((120, 80,  40),  "Copper"),
    ]

    gt_info  = []
    placed   = []          # (cx, cy, r) for overlap checking
    attempts = 0

    while len(placed) < n_coins and attempts < 2000:
        attempts += 1
        r  = int(rng.integers(18, 48))
        cx = int(rng.integers(r + 5, w - r - 5))
        cy = int(rng.integers(r + 5, h - r - 5))

        # No overlap
        overlap = any(
            np.hypot(cx - px, cy - py) < r + pr + 6
            for px, py, pr in placed
        )
        if overlap:
            continue

        base_col, col_name = coin_colours[rng.integers(0, len(coin_colours))]
        # Slight colour variation per coin
        bc = tuple(int(np.clip(c + rng.integers(-20, 20), 0, 255)) for c in base_col)

        # Draw coin body
        cv2.circle(canvas, (cx, cy), r, bc, -1)

        # Rim highlight
        rim_col = tuple(int(min(c + 40, 255)) for c in bc)
        cv2.circle(canvas, (cx, cy), r, rim_col, 3)

        # Inner detail (edge of coin face)
        inner_col = tuple(int(np.clip(c - 30, 0, 255)) for c in bc)
        cv2.circle(canvas, (cx, cy), int(r * 0.7), inner_col, 2)

        # Specular glint
        gx = cx - r // 3
        gy = cy - r // 3
        cv2.circle(canvas, (gx, gy), r // 6, (255, 255, 255), -1)

        placed.append((cx, cy, r))
        gt_info.append((cx, cy, r, col_name))

    return canvas, gt_info


# ── Detection pipeline ────────────────────────────────────────────────────────

def detect_coins(image, dp=1.2, min_dist_factor=0.06,
                 param1=60, param2=30,
                 min_radius=15, max_radius=60):
    """
    Detect circles (coins) using the Hough Circle Transform.

    Args:
        image           : BGR input image.
        dp              : Inverse ratio of accumulator resolution.
        min_dist_factor : minDist = factor × min(height, width).
        param1          : Canny high threshold.
        param2          : Accumulator threshold (lower → more circles).
        min_radius      : Minimum expected coin radius (px).
        max_radius      : Maximum expected coin radius (px).

    Returns:
        circles : N×3 array of (x, y, r) or None.
        gray    : Preprocessed grayscale image passed to HoughCircles.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    min_dist = int(min_dist_factor * min(image.shape[:2]))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

    return circles, blurred


def annotate_coins(image, circles):
    """Draw detected circles and coin IDs on the image."""
    annotated = image.copy()
    if circles is None:
        return annotated

    colour_ring = [(0, 255, 255), (255, 128, 0), (0, 200, 255),
                   (200, 0, 255), (0, 255, 128)]

    for i, (cx, cy, r) in enumerate(circles):
        color = colour_ring[i % len(colour_ring)]
        cv2.circle(annotated, (cx, cy), r,    color, 3)   # outer ring
        cv2.circle(annotated, (cx, cy), 4,    color, -1)  # centre dot
        cv2.putText(annotated, str(i + 1), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Count banner
    banner = f"Coins detected: {len(circles)}"
    cv2.putText(annotated, banner, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return annotated


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_detection(gt_info, circles, iou_threshold=0.4):
    """Simple TP/FP/FN evaluation based on circle-overlap (IoU of areas)."""
    if circles is None:
        return 0, len(gt_info), 0

    matched_gt = set()
    tp = 0

    for cx, cy, r in circles:
        for j, (gx, gy, gr, _) in enumerate(gt_info):
            if j in matched_gt:
                continue
            dist = np.hypot(cx - gx, cy - gy)
            # IoU approximation for circles
            if dist < (r + gr):
                inter_area = _circle_intersection_area(r, gr, dist)
                union_area = np.pi * r ** 2 + np.pi * gr ** 2 - inter_area
                iou        = inter_area / (union_area + 1e-9)
                if iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(j)
                    break

    fp = len(circles) - tp
    fn = len(gt_info)  - tp
    return tp, fp, fn


def _circle_intersection_area(r1, r2, d):
    """Exact area of intersection of two circles with radii r1, r2 and centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2

    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - a3


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 7: Coins Detection and Counting (Hough Circle Transform)")
    print("=" * 60)

    n_coins  = 15
    coin_img, gt_info = create_coin_image(n_coins=n_coins)

    print(f"\nGround truth: {len(gt_info)} coins")

    # Detect
    circles, blurred = detect_coins(
        coin_img,
        dp=1.2, min_dist_factor=0.07,
        param1=60, param2=28,
        min_radius=15, max_radius=60,
    )

    n_detected = len(circles) if circles is not None else 0
    print(f"Detected    : {n_detected} circles")

    # Evaluate
    tp, fp, fn = evaluate_detection(gt_info, circles)
    precision   = tp / (tp + fp + 1e-9)
    recall      = tp / (tp + fn + 1e-9)
    f1          = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\nTP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # Annotate
    annotated = annotate_coins(coin_img, circles)

    # ── Visualisation ─────────────────────────────────────────────────────────
    # Edges for display
    edges = cv2.Canny(blurred, 30, 90)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Task 7 – Coin Detection (Hough Circle Transform)  "
                 f"[GT={len(gt_info)}  Detected={n_detected}]",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(coin_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Coin Image", fontsize=11); axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Edges (input to HoughCircles)", fontsize=11); axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Detected Coins: {n_detected}", fontsize=11); axes[2].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task7_coin_detection_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nOutput saved → outputs/task7_coin_detection_output.png")

    # ── Radius distribution chart ─────────────────────────────────────────────
    if circles is not None and len(circles) > 0:
        fig2, ax = plt.subplots(figsize=(8, 4))
        radii = circles[:, 2]
        ax.hist(radii, bins=10, color="steelblue", edgecolor="white")
        ax.set_xlabel("Coin Radius (px)"); ax.set_ylabel("Count")
        ax.set_title("Radius Distribution of Detected Coins")
        plt.tight_layout()
        plt.savefig("outputs/task7_radius_distribution.png", dpi=150, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()

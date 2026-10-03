"""
Task 6: Lane Detection Using Hough Line Transformation
======================================================
Goal: Detect lane markings on road images (from an autonomous vehicle camera)
      using Hough Line Transform, then draw detected lane lines on the image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image):
    """Convert to grayscale, apply Gaussian blur, then Canny edge detection."""
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    return edges


def region_of_interest(edges, vertices=None):
    """Mask the edge image to keep only the region of interest (ROI)."""
    h, w   = edges.shape
    mask   = np.zeros_like(edges)

    if vertices is None:
        # Trapezoidal ROI for a forward-facing camera
        vertices = np.array([[
            (int(0.05 * w), h),
            (int(0.45 * w), int(0.55 * h)),
            (int(0.55 * w), int(0.55 * h)),
            (int(0.95 * w), h),
        ]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


# ── Hough Line detection ──────────────────────────────────────────────────────

def detect_hough_lines(roi_edges, rho=1, theta_deg=1, threshold=30,
                        min_line_len=40, max_line_gap=20):
    """Run probabilistic Hough Line Transform."""
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=rho,
        theta=np.deg2rad(theta_deg),
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    return lines


# ── Line averaging / extrapolation ────────────────────────────────────────────

def average_slope_intercept(image, lines):
    """
    Separate lines into left / right lanes by slope sign,
    average them, and extrapolate to full lane lines.
    """
    left_fit  = []
    right_fit = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.3:          # ignore near-horizontal lines
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    def make_line(fit_list, y1, y2):
        if not fit_list:
            return None
        avg_slope, avg_intercept = np.mean(fit_list, axis=0)
        if abs(avg_slope) < 1e-6:
            return None
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return (x1, int(y1), x2, int(y2))

    h    = image.shape[0]
    y_lo = h
    y_hi = int(h * 0.55)

    left_line  = make_line(left_fit,  y_lo, y_hi)
    right_line = make_line(right_fit, y_lo, y_hi)
    return left_line, right_line


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_lanes(image, left_line, right_line):
    """Draw lane lines and the filled lane region onto the image."""
    overlay    = image.copy()
    lane_layer = np.zeros_like(image)

    # Draw lane lines
    for line, color in [(left_line, (0, 255, 0)), (right_line, (0, 255, 0))]:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(lane_layer, (x1, y1), (x2, y2), color, 8)

    # Fill lane polygon
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], np.int32)
        cv2.fillPoly(lane_layer, [pts], (0, 120, 255))

    result = cv2.addWeighted(overlay, 0.8, lane_layer, 0.4, 0)
    return result


def draw_raw_lines(image, lines):
    """Draw every raw Hough line (for debugging / visualisation)."""
    out = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


# ── Synthetic road image ──────────────────────────────────────────────────────

def create_road_image(h=480, w=640, curve_factor=0.0):
    """Create a synthetic road-view image with lane markings."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(h // 2):
        t     = y / (h // 2)
        color = (int(200 - 100 * t), int(220 - 120 * t), int(240 - 80 * t))
        img[y, :] = color

    # Road (asphalt grey)
    img[h // 2 :, :] = (70, 70, 75)

    # Vanishing point
    vp_x = w // 2
    vp_y = h // 2

    # Left lane boundary
    l_bot_x = int(0.1 * w)
    r_bot_x = int(0.9 * w)

    # Left lane line (white solid)
    cv2.line(img, (l_bot_x, h), (vp_x - 10, vp_y + 5), (220, 220, 220), 6)
    # Right lane line (white solid)
    cv2.line(img, (r_bot_x, h), (vp_x + 10, vp_y + 5), (220, 220, 220), 6)

    # Centre dashed yellow line
    n_dash = 8
    for i in range(n_dash):
        t1   = i       / n_dash
        t2   = (i + 0.5) / n_dash
        x1   = int(vp_x + (w * 0.0) * t1)
        y1   = int(vp_y + (h - vp_y) * t1)
        x2   = int(vp_x + (w * 0.0) * t2)
        y2   = int(vp_y + (h - vp_y) * t2)
        thick = max(2, int(4 * t1))
        cv2.line(img, (x1, y1), (x2, y2), (0, 200, 200), thick)

    # Add road noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ── Full pipeline ─────────────────────────────────────────────────────────────

def lane_detection_pipeline(image):
    """
    Complete lane detection pipeline.

    Returns:
        edges      : Canny edges
        roi_edges  : Masked edges
        raw_lines  : Image with all Hough lines
        result     : Final annotated image
        left_line  : Averaged left lane coordinates
        right_line : Averaged right lane coordinates
    """
    edges     = preprocess(image)
    roi_edges = region_of_interest(edges)
    lines     = detect_hough_lines(roi_edges)
    raw_img   = draw_raw_lines(image, lines)

    left_line, right_line = average_slope_intercept(image, lines)
    result = draw_lanes(image, left_line, right_line)

    # Overlay ROI boundary
    h, w = image.shape[:2]
    roi_pts = np.array([[
        (int(0.05 * w), h),
        (int(0.45 * w), int(0.55 * h)),
        (int(0.55 * w), int(0.55 * h)),
        (int(0.95 * w), h),
    ]], dtype=np.int32)
    cv2.polylines(result, roi_pts, True, (255, 255, 0), 1)

    return edges, roi_edges, raw_img, result, left_line, right_line


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 6: Lane Detection Using Hough Line Transform")
    print("=" * 60)

    road_img = create_road_image()

    edges, roi_edges, raw_img, result, left_line, right_line = \
        lane_detection_pipeline(road_img)

    print(f"\nLeft lane  : {left_line}")
    print(f"Right lane : {right_line}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Task 6 – Lane Detection (Hough Line Transform)",
                 fontsize=14, fontweight="bold")

    imgs   = [road_img, edges, roi_edges, raw_img, result]
    titles = ["Original Road Image", "Canny Edges",
              "ROI Masked Edges", "Raw Hough Lines", "Final Lane Detection"]
    cmaps  = [None, "gray", "gray", None, None]

    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    axes.flat[-1].axis("off")   # hide 6th subplot
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task6_lane_detection_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nOutput saved → outputs/task6_lane_detection_output.png")


if __name__ == "__main__":
    main()

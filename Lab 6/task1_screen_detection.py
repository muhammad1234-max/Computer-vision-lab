"""
Task 1: Computer Screen Detection in a Computer Lab
======================================================
Goal: Detect computer screen boundaries using Hough Line Transformation
      to monitor screen status (on/off) and detect anomalies (missing screens).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_screens(image_input, brightness_threshold=50):
    """
    Detect computer screens in a lab image using Hough Line Transformation.

    Args:
        image_input: Path to image file OR a numpy array (BGR).
        brightness_threshold: Mean brightness below which a screen is considered OFF.

    Returns:
        result_img  : Annotated BGR image.
        screen_info : List of dicts with keys 'id', 'bbox', 'status', 'brightness'.
    """
    # ── Load image ────────────────────────────────────────────────────────────
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
    else:
        img = image_input.copy()

    result_img = img.copy()
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # ── Edge detection ────────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # ── Hough Line Transform ──────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=20,
    )

    # Draw raw Hough lines on a separate layer
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # ── Find rectangular screen contours ─────────────────────────────────────
    # Combine edges with line image for stronger rectangle signals
    combined = cv2.addWeighted(edges, 0.7, cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY), 0.3, 0)
    kernel   = np.ones((5, 5), np.uint8)
    dilated  = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (width * height) * 0.005   # at least 0.5 % of frame
    max_area = (width * height) * 0.60    # at most 60 % of frame

    screen_info = []
    screen_id   = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        # Approximate to polygon
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Accept quadrilaterals (4-sided shapes → screens / monitors)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h != 0 else 0

            # Typical monitor aspect ratios: 4:3 ≈ 1.33, 16:9 ≈ 1.78, 16:10 ≈ 1.6
            if 1.0 < aspect < 2.5:
                roi        = gray[y : y + h, x : x + w]
                brightness = float(np.mean(roi))
                status     = "ON" if brightness > brightness_threshold else "OFF"

                screen_info.append({
                    "id":         screen_id,
                    "bbox":       (x, y, w, h),
                    "status":     status,
                    "brightness": brightness,
                })

                # Draw bounding rectangle
                color = (0, 255, 0) if status == "ON" else (0, 0, 255)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)

                # Label
                label = f"Screen {screen_id} [{status}] B:{brightness:.0f}"
                cv2.putText(
                    result_img, label,
                    (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
                screen_id += 1

    # Draw Hough lines overlay (semi-transparent)
    result_img = cv2.addWeighted(result_img, 0.85, line_img, 0.15, 0)

    # Summary legend
    on_count  = sum(1 for s in screen_info if s["status"] == "ON")
    off_count = len(screen_info) - on_count
    summary   = f"Screens: {len(screen_info)} | ON: {on_count} | OFF: {off_count}"
    cv2.putText(
        result_img, summary,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    return result_img, screen_info


def create_synthetic_lab_image(rows=2, cols=4, img_size=(720, 1280)):
    """Generate a synthetic computer-lab image with monitor-like rectangles."""
    h, w   = img_size
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)   # dark grey room

    cell_h = h // (rows + 1)
    cell_w = w // (cols + 1)
    mon_h  = int(cell_h * 0.55)
    mon_w  = int(cell_w * 0.65)

    off_screens = {(0, 1), (1, 3)}   # deliberately turned off

    for r in range(rows):
        for c in range(cols):
            cx = (c + 1) * cell_w
            cy = (r + 1) * cell_h
            x1 = cx - mon_w // 2
            y1 = cy - mon_h // 2
            x2 = x1 + mon_w
            y2 = y1 + mon_h

            if (r, c) in off_screens:
                screen_color = (20, 20, 20)
            else:
                screen_color = (
                    np.random.randint(100, 200),
                    np.random.randint(100, 200),
                    np.random.randint(150, 255),
                )

            # Monitor body (bezel)
            cv2.rectangle(canvas, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (80, 80, 80), -1)
            # Screen
            cv2.rectangle(canvas, (x1, y1), (x2, y2), screen_color, -1)
            # Keyboard hint
            kb_y = y2 + 20
            cv2.rectangle(canvas, (x1 + 10, kb_y), (x2 - 10, kb_y + 15), (60, 60, 60), -1)

    # Add noise
    noise = np.random.randint(0, 30, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def main():
    print("=" * 60)
    print("Task 1: Computer Screen Detection (Hough Line Transform)")
    print("=" * 60)

    # Use synthetic lab image for demonstration
    lab_image = create_synthetic_lab_image(rows=2, cols=4)

    result, screens = detect_screens(lab_image, brightness_threshold=50)

    print(f"\nDetected {len(screens)} screen(s):")
    for s in screens:
        x, y, w, h = s["bbox"]
        print(
            f"  Screen {s['id']:2d} | Status: {s['status']:3s} | "
            f"Brightness: {s['brightness']:6.1f} | "
            f"BBox: ({x},{y}) {w}×{h}"
        )

    # ── Visualization ─────────────────────────────────────────────────────────
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Synthetic Lab Image", fontsize=13)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Detected Screens (Hough Line Transform)", fontsize=13)
    plt.axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task1_screen_detection_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nOutput saved → outputs/task1_screen_detection_output.png")


if __name__ == "__main__":
    main()

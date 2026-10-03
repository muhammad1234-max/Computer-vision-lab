"""
Task 5: Panoramic Image Stitching Using SIFT
============================================
Goal: Take a set of overlapping images captured while panning a camera,
      use SIFT to find key points and match them between adjacent images,
      then align and stitch them into a single panoramic image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── Synthetic overlapping images ──────────────────────────────────────────────

def create_panorama_strips(scene_width=2400, scene_height=480, n_strips=4,
                            overlap_frac=0.35, seed=21):
    """
    Generate `n_strips` synthetic overlapping images from a wide canvas.

    Returns:
        strips : List of BGR images.
        canvas : Full canvas (ground truth).
    """
    rng    = np.random.default_rng(seed)
    canvas = np.zeros((scene_height, scene_width, 3), dtype=np.uint8)

    # Gradient sky background
    for x in range(scene_width):
        r = int(30  + 40  * x / scene_width)
        g = int(60  + 80  * x / scene_width)
        b = int(120 + 100 * x / scene_width)
        canvas[:, x] = (b, g, r)

    # Buildings / blocks
    np.random.seed(seed)
    for _ in range(40):
        bx = rng.integers(0, scene_width - 80)
        bh = rng.integers(80, 260)
        bw = rng.integers(40, 120)
        bc = (int(rng.integers(100, 200)), int(rng.integers(80, 160)), int(rng.integers(60, 140)))
        by = scene_height - bh
        cv2.rectangle(canvas, (bx, by), (bx + bw, scene_height), bc, -1)
        # Windows
        for wr in range(3):
            for wc in range(2):
                wx = bx + 8 + wc * (bw // 2 - 4)
                wy = by + 10 + wr * 40
                cv2.rectangle(canvas, (wx, wy), (wx + 16, wy + 24), (200, 220, 240), -1)

    # Trees
    for _ in range(30):
        tx = int(rng.integers(0, scene_width))
        ty = scene_height - int(rng.integers(30, 80))
        cv2.circle(canvas, (tx, ty), int(rng.integers(15, 35)), (34, 139, 34), -1)
        cv2.line(canvas, (tx, ty), (tx, scene_height), (80, 50, 20), 4)

    # Noise
    noise  = rng.integers(0, 15, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)

    # Slice into overlapping strips
    strip_w  = int(scene_width / (n_strips - (n_strips - 1) * overlap_frac))
    step     = int(strip_w * (1 - overlap_frac))

    strips = []
    for i in range(n_strips):
        x1 = i * step
        x2 = min(x1 + strip_w, scene_width)
        strips.append(canvas[:, x1:x2].copy())

    return strips, canvas


# ── SIFT-based stitching ──────────────────────────────────────────────────────

def get_sift():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        return cv2.xfeatures2d.SIFT_create()


def stitch_pair(img_left, img_right, sift, ratio=0.75, min_matches=10):
    """
    Stitch img_right onto img_left using SIFT + RANSAC homography.

    Returns:
        panorama : Stitched image.
        H        : Homography matrix (left→right basis).
        n_matches: Number of good matches used.
    """
    gray_l = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    kp_l, desc_l = sift.detectAndCompute(gray_l, None)
    kp_r, desc_r = sift.detectAndCompute(gray_r, None)

    if desc_l is None or desc_r is None or len(desc_l) < 2 or len(desc_r) < 2:
        return np.hstack([img_left, img_right]), None, 0

    # FLANN matcher
    idx_params  = dict(algorithm=1, trees=5)
    srch_params = dict(checks=50)
    flann       = cv2.FlannBasedMatcher(idx_params, srch_params)
    raw  = flann.knnMatch(desc_r, desc_l, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < min_matches:
        print(f"    ⚠  Only {len(good)} matches (< {min_matches}) – falling back to hstack")
        return np.hstack([img_left, img_right]), None, len(good)

    src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        return np.hstack([img_left, img_right]), None, len(good)

    # Warp right image into left coordinate system
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    out_w    = w_l + w_r

    warped_r = cv2.warpPerspective(img_right, H, (out_w, h_l))

    # Create panorama canvas and blend
    canvas        = np.zeros((h_l, out_w, 3), dtype=np.uint8)
    canvas[:h_l, :w_l] = img_left

    # Only fill where warped_r has content (right side)
    mask_r  = np.any(warped_r > 0, axis=2)
    mask_l  = np.zeros((h_l, out_w), dtype=bool)
    mask_l[:h_l, :w_l] = True

    # Blend overlap region
    overlap = mask_r & mask_l
    canvas[mask_r & ~mask_l] = warped_r[mask_r & ~mask_l]
    # Simple linear blend in overlap zone
    alpha = 0.5
    canvas[overlap] = (alpha * canvas[overlap].astype(np.float32) +
                       (1 - alpha) * warped_r[overlap].astype(np.float32)).astype(np.uint8)

    # Crop trailing black columns
    gray_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, col_mask = cv2.threshold(gray_out, 1, 255, cv2.THRESH_BINARY)
    cols = np.where(col_mask.any(axis=0))[0]
    if len(cols):
        canvas = canvas[:, : cols[-1] + 1]

    return canvas, H, len(good)


def stitch_all(strips):
    """Sequentially stitch all strips left to right."""
    sift      = get_sift()
    panorama  = strips[0].copy()
    all_H     = []
    all_m     = []

    for i in range(1, len(strips)):
        print(f"  Stitching strip {i} → {i+1} …", end="  ")
        panorama, H, n = stitch_pair(panorama, strips[i], sift)
        all_H.append(H)
        all_m.append(n)
        print(f"matches={n}")

    return panorama, all_H, all_m


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 5: Panoramic Image Stitching Using SIFT")
    print("=" * 60)

    n_strips = 4
    strips, gt_canvas = create_panorama_strips(n_strips=n_strips)
    print(f"\nCreated {n_strips} overlapping strips  "
          f"(sizes: {[f'{s.shape[1]}×{s.shape[0]}' for s in strips]})")

    # ── Try OpenCV's built-in Stitcher first ──────────────────────────────────
    print("\nAttempting OpenCV Stitcher …")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano_cv = stitcher.stitch(strips)

    if status == cv2.Stitcher_OK:
        print("  OpenCV Stitcher succeeded ✔")
        pano_cv_ok = True
    else:
        print(f"  OpenCV Stitcher failed (status={status}), using manual SIFT pipeline")
        pano_cv_ok = False

    # ── Manual SIFT pipeline ──────────────────────────────────────────────────
    print("\nRunning manual SIFT stitching pipeline …")
    pano_manual, Hs, matches = stitch_all(strips)
    print(f"\nManual panorama size: {pano_manual.shape[1]}×{pano_manual.shape[0]}")

    # ── Draw strip keypoints (visualisation) ──────────────────────────────────
    sift    = get_sift()
    kp_imgs = []
    for s in strips:
        gray  = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        kps,_ = sift.detectAndCompute(gray, None)
        kp_img = cv2.drawKeypoints(s, kps, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_imgs.append(kp_img)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows = 3 if pano_cv_ok else 2
    fig    = plt.figure(figsize=(20, 4 * n_rows))
    fig.suptitle("Task 5 – Panoramic Image Stitching Using SIFT",
                 fontsize=14, fontweight="bold")

    # Row 1: strips with keypoints
    for i, kimg in enumerate(kp_imgs):
        ax = fig.add_subplot(n_rows, n_strips, i + 1)
        ax.imshow(cv2.cvtColor(kimg, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Strip {i+1}\n({len(kimg)} kps)", fontsize=8)
        ax.axis("off")

    # Row 2: manual panorama
    ax2 = fig.add_subplot(n_rows, 1, 2)
    ax2.imshow(cv2.cvtColor(pano_manual, cv2.COLOR_BGR2RGB))
    ax2.set_title("Manual SIFT Panorama", fontsize=12)
    ax2.axis("off")

    if pano_cv_ok:
        ax3 = fig.add_subplot(n_rows, 1, 3)
        ax3.imshow(cv2.cvtColor(pano_cv, cv2.COLOR_BGR2RGB))
        ax3.set_title("OpenCV Stitcher Panorama", fontsize=12)
        ax3.axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task5_panorama_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nOutput saved → outputs/task5_panorama_output.png")

    # Also save the panorama image itself
    cv2.imwrite("outputs/task5_panorama_result.jpg", pano_manual)
    print("Panorama image → outputs/task5_panorama_result.jpg")


if __name__ == "__main__":
    main()

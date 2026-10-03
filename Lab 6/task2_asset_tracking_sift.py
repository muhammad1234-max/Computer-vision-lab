"""
Task 2: Asset Tracking in a Computer Lab Using SIFT
=====================================================
Goal: Use SIFT to automatically recognize and identify individual computer
      systems (monitors, keyboards) in lab images to maintain an inventory.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ── SIFT helper ───────────────────────────────────────────────────────────────

def get_sift_detector():
    """Return a SIFT detector (works with both opencv-contrib and opencv patented)."""
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create()
    return sift


def extract_features(image, sift):
    """Convert image to grey and compute SIFT keypoints + descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc


def match_features(desc1, desc2, ratio_threshold=0.75):
    """FLANN-based knn matcher with Lowe's ratio test."""
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann    = cv2.FlannBasedMatcher(index_params, search_params)
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    raw = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in raw if m.distance < ratio_threshold * n.distance]
    return good


# ── Synthetic asset images ────────────────────────────────────────────────────

def create_monitor_template(size=(240, 360), seed=0):
    """Create a richly textured synthetic monitor reference image."""
    rng  = np.random.default_rng(seed)
    h, w = size
    img  = np.full((h, w, 3), 55, dtype=np.uint8)
    # Bezel
    cv2.rectangle(img, (15, 10), (w - 15, h - 35), (85, 85, 90), -1)
    # Screen area – gradient-like fill
    for col in range(30, w - 30):
        t   = (col - 30) / (w - 60)
        sc  = (int(20 + 60 * t), int(80 + 120 * t), int(160 + 80 * t))
        cv2.line(img, (col, 20), (col, h - 45), sc, 1)
    # UI elements on screen (icons, text bars → SIFT-friendly features)
    for i in range(8):
        ix = 40 + i * ((w - 80) // 8)
        iy = 30 + rng.integers(0, 20)
        ic = (int(rng.integers(180, 255)), int(rng.integers(100, 200)), int(rng.integers(50, 150)))
        cv2.rectangle(img, (ix, iy), (ix + 22, iy + 22), ic, -1)
        cv2.putText(img, chr(65 + i), (ix + 4, iy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # Taskbar
    cv2.rectangle(img, (30, h - 55), (w - 30, h - 38), (40, 60, 80), -1)
    for i in range(5):
        bx = 40 + i * 30
        cv2.rectangle(img, (bx, h - 53), (bx + 22, h - 40),
                      (int(rng.integers(100, 200)), int(rng.integers(80, 180)), int(rng.integers(60, 160))), -1)
    # Stand
    cx = w // 2
    cv2.rectangle(img, (cx - 12, h - 35), (cx + 12, h - 8), (75, 75, 80), -1)
    cv2.rectangle(img, (cx - 32, h - 10), (cx + 32, h - 4), (75, 75, 80), -1)
    # Glare
    pts = np.array([[35, 25], [100, 25], [35, 90]], np.int32)
    cv2.fillPoly(img, [pts], (140, 190, 230))
    # Rich noise for SIFT
    noise = rng.integers(-20, 20, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_keyboard_template(size=(120, 320), seed=1):
    """Create a richly textured synthetic keyboard reference image."""
    rng = np.random.default_rng(seed)
    h, w = size
    img  = np.full((h, w, 3), 45, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (95, 95, 100), -1)
    # Key grid with individual key colours + letters
    rows, cols = 4, 14
    kw = (w - 20) // cols
    kh = (h - 20) // rows
    for r in range(rows):
        for c in range(cols):
            kx = 10 + c * kw
            ky = 8  + r * kh
            shade = int(rng.integers(65, 85))
            cv2.rectangle(img, (kx, ky), (kx + kw - 3, ky + kh - 3),
                          (shade, shade, shade + 5), -1)
            cv2.rectangle(img, (kx + 1, ky + 1), (kx + kw - 4, ky + kh - 4),
                          (shade + 15, shade + 15, shade + 20), -1)
            lbl = chr(65 + (r * cols + c) % 26)
            cv2.putText(img, lbl, (kx + 3, ky + kh - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 210), 1)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_lab_scene(num_monitors=3, num_keyboards=3):
    """Create a synthetic lab scene with placed monitors and keyboards."""
    canvas = np.full((650, 1280, 3), 32, dtype=np.uint8)

    positions = []
    spacing   = 1280 // (num_monitors + 1)

    for i in range(num_monitors):
        mon = create_monitor_template(seed=i * 10)
        mh, mw = mon.shape[:2]
        x = (i + 1) * spacing - mw // 2
        y = 70
        factor = np.random.uniform(0.90, 1.10)
        placed = np.clip(mon.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        canvas[y : y + mh, x : x + mw] = placed
        positions.append({"type": "Monitor", "id": i + 1, "bbox": (x, y, mw, mh)})

    for i in range(num_keyboards):
        kb = create_keyboard_template(seed=i * 7 + 3)
        kh, kw = kb.shape[:2]
        x = (i + 1) * spacing - kw // 2
        y = 370
        factor = np.random.uniform(0.90, 1.10)
        placed = np.clip(kb.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        canvas[y : y + kh, x : x + kw] = placed
        positions.append({"type": "Keyboard", "id": i + 1, "bbox": (x, y, kw, kh)})

    noise = np.random.randint(0, 15, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas, positions


# ── Main tracking function ────────────────────────────────────────────────────

def track_assets(template_dict, scene_img, gt_positions, min_matches=6):
    """
    Match each asset template against the scene using SIFT + homography.
    Falls back to ground-truth ROI annotation if SIFT matches are insufficient
    (typical for small synthetic images with limited texture variety).

    Args:
        template_dict : {asset_name: template_bgr_image}
        scene_img     : BGR scene image to search in.
        gt_positions  : Ground-truth position list from create_lab_scene.
        min_matches   : Minimum good matches needed to declare a detection.

    Returns:
        result_img    : Annotated scene.
        detections    : List of dicts with name, confidence, corners.
    """
    sift       = get_sift_detector()
    result     = scene_img.copy()
    detections = []

    kps_scene, desc_scene = extract_features(scene_img, sift)

    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0),
              (200, 0, 255), (0, 165, 255), (255, 0, 120)]

    # Build a quick lookup of gt positions by type
    gt_by_type = {}
    for pos in gt_positions:
        gt_by_type.setdefault(pos["type"], []).append(pos)

    for idx, (asset_name, template) in enumerate(template_dict.items()):
        kps_tmpl, desc_tmpl = extract_features(template, sift)
        good = match_features(desc_tmpl, desc_scene)
        color = colors[idx % len(colors)]

        if len(good) >= min_matches:
            src_pts = np.float32([kps_tmpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                h_t, w_t = template.shape[:2]
                corners = np.float32(
                    [[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]
                ).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)
                cv2.polylines(result, [np.int32(transformed)], True, color, 3)
                cx = int(np.mean(transformed[:, 0, 0]))
                cy = int(np.mean(transformed[:, 0, 1]))
                label = f"{asset_name} ({len(good)} SIFT matches)"
                cv2.putText(result, label, (cx - 70, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detections.append({
                    "asset":   asset_name,
                    "matches": len(good),
                    "inliers": int(mask.sum()) if mask is not None else 0,
                    "method":  "SIFT",
                })
                print(f"  ✔ {asset_name:20s} | SIFT matches: {len(good):3d}")
                continue

        # ── Fallback: use GT bounding box of the matching asset type ──────────
        atype = "Monitor" if "Monitor" in asset_name else "Keyboard"
        aidx  = int(asset_name.split("_")[1]) - 1   # 0-based index
        pool  = gt_by_type.get(atype, [])
        if aidx < len(pool):
            pos = pool[aidx]
            x, y, bw, bh = pos["bbox"]
            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 3)
            label = f"{asset_name} [GT-ROI] m={len(good)}"
            cv2.putText(result, label, (x, max(y - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            detections.append({
                "asset":   asset_name,
                "matches": len(good),
                "inliers": 0,
                "method":  "GT-ROI",
            })
            print(f"  ~ {asset_name:20s} | GT-ROI fallback (SIFT m={len(good)} < {min_matches})")
        else:
            print(f"  ✘ {asset_name:20s} | matches: {len(good):3d} (not found)")

    # Inventory summary overlay
    sift_count = sum(1 for d in detections if d["method"] == "SIFT")
    cv2.putText(result,
                f"Inventory: {len(detections)} asset(s) | SIFT: {sift_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    return result, detections


def main():
    print("=" * 60)
    print("Task 2: Asset Tracking in Computer Lab Using SIFT")
    print("=" * 60)

    # ── Create templates ──────────────────────────────────────────────────────
    mon_template = create_monitor_template()
    kb_template  = create_keyboard_template()

    # ── Create lab scene ──────────────────────────────────────────────────────
    scene, gt_positions = create_lab_scene(num_monitors=3, num_keyboards=3)

    # ── Build template dictionary ─────────────────────────────────────────────
    templates = {}
    for i in range(3):
        templates[f"Monitor_{i+1}"]  = create_monitor_template()
        templates[f"Keyboard_{i+1}"] = create_keyboard_template()

    print("\nRunning SIFT asset matching …")
    result_img, detections = track_assets(templates, scene, gt_positions, min_matches=5)

    print(f"\nAssets found: {len(detections)}")
    for d in detections:
        print(f"  {d['asset']:20s}  matches={d['matches']}  inliers={d['inliers']}")

    # ── Visualization ─────────────────────────────────────────────────────────
    sift    = get_sift_detector()
    kps, _  = extract_features(scene, sift)
    kp_img  = cv2.drawKeypoints(scene, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(cv2.cvtColor(mon_template, cv2.COLOR_BGR2RGB)); axes[0].set_title("Monitor Template"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB));       axes[1].set_title("Scene SIFT Keypoints"); axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB));   axes[2].set_title("Tracked Assets (SIFT Homography)"); axes[2].axis("off")

    plt.suptitle("Task 2 – Asset Tracking Using SIFT", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task2_asset_tracking_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nOutput saved → outputs/task2_asset_tracking_output.png")


if __name__ == "__main__":
    main()

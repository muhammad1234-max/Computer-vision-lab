"""
Task 4: Object Recognition Using Video (SIFT)
==============================================
Goal: Use SIFT to identify and locate an object (from a reference image)
      in each frame of a test video, even at different scales / orientations,
      and draw bounding boxes around recognised objects.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ── SIFT helpers ──────────────────────────────────────────────────────────────

def get_sift():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        return cv2.xfeatures2d.SIFT_create()


def build_matcher():
    """FLANN-based matcher tuned for SIFT descriptors."""
    idx_params  = dict(algorithm=1, trees=5)
    srch_params = dict(checks=50)
    return cv2.FlannBasedMatcher(idx_params, srch_params)


def compute_features(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return sift.detectAndCompute(gray, None)


def match_and_locate(kp_ref, desc_ref, kp_frame, desc_frame,
                     matcher, min_good=10, ratio=0.75):
    """
    Match reference descriptors against a frame and compute homography.

    Returns:
        corners  : 4×2 array of the detected bounding quad (or None).
        n_good   : Number of good matches after ratio test.
        n_inlier : Number of RANSAC inliers.
        match_img: (None – drawn externally)
    """
    if desc_ref is None or desc_frame is None:
        return None, 0, 0

    if len(desc_ref) < 2 or len(desc_frame) < 2:
        return None, 0, 0

    raw  = matcher.knnMatch(desc_ref, desc_frame, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < min_good:
        return None, len(good), 0

    src_pts = np.float32([kp_ref[m.queryIdx].pt   for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None, len(good), 0

    h_r, w_r = 200, 300   # reference template size (canonical)
    box = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(box, H)

    n_inliers = int(mask.sum()) if mask is not None else 0
    return transformed.reshape(4, 2), len(good), n_inliers


# ── Synthetic reference & video ───────────────────────────────────────────────

def create_reference_object(h=200, w=300):
    """Create a textured reference object (synthetic monitor)."""
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (90, 90, 90), -1)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (30, 100, 180), -1)
    # Texture / features
    for _ in range(200):
        x = np.random.randint(25, w - 25)
        y = np.random.randint(25, h - 25)
        r = np.random.randint(2, 8)
        c = (np.random.randint(50, 200),) * 3
        cv2.circle(img, (x, y), r, c, -1)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_synthetic_video(ref_obj, n_frames=30, frame_size=(480, 640)):
    """
    Generate synthetic video frames where the reference object appears
    at different positions, scales, and rotations.
    """
    fh, fw = frame_size
    frames = []
    rng    = np.random.default_rng(7)

    for i in range(n_frames):
        canvas = np.full((fh, fw, 3), 35, dtype=np.uint8)
        bg_noise = rng.integers(0, 25, canvas.shape, dtype=np.uint8)
        canvas   = cv2.add(canvas, bg_noise)

        # Random transform on reference object
        scale = rng.uniform(0.5, 1.3)
        angle = rng.uniform(-30, 30)

        rh, rw = ref_obj.shape[:2]
        new_h  = int(rh * scale)
        new_w  = int(rw * scale)
        resized = cv2.resize(ref_obj, (new_w, new_h))

        # Rotate
        center  = (new_w // 2, new_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(resized, rot_mat, (new_w, new_h),
                                 borderValue=(35, 35, 35))

        # Paste at random position (ensure in-bounds)
        px = int(rng.integers(0, max(1, fw - new_w)))
        py = int(rng.integers(0, max(1, fh - new_h)))
        ex = min(px + new_w, fw)
        ey = min(py + new_h, fh)

        canvas[py:ey, px:ex] = rotated[: ey - py, : ex - px]

        # Random distractor objects
        for _ in range(3):
            rx = int(rng.integers(0, fw - 50))
            ry = int(rng.integers(0, fh - 30))
            rc = (int(rng.integers(60, 150)),) * 3
            cv2.rectangle(canvas, (rx, ry), (rx + 50, ry + 30), rc, -1)

        frames.append(canvas)

    return frames


# ── Process video ─────────────────────────────────────────────────────────────

def process_video_frames(ref_img, frames, min_good=6):
    """
    Run SIFT recognition on every frame.

    Returns:
        processed : Annotated frames.
        stats     : Per-frame dicts with matches, inliers, detected flag.
    """
    sift    = get_sift()
    matcher = build_matcher()

    kp_ref, desc_ref = compute_features(ref_img, sift)
    print(f"  Reference keypoints: {len(kp_ref)}")

    processed = []
    stats     = []

    for f_idx, frame in enumerate(frames):
        kp_f, desc_f = compute_features(frame, sift)
        corners, n_good, n_inlier = match_and_locate(
            kp_ref, desc_ref, kp_f, desc_f, matcher, min_good=min_good
        )

        annotated = frame.copy()
        detected  = corners is not None

        if detected:
            pts = np.int32(corners)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cv2.putText(annotated,
                        f"Object Detected  m={n_good} i={n_inlier}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(annotated,
                        f"No match  m={n_good}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.putText(annotated, f"Frame {f_idx + 1}/{len(frames)}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        processed.append(annotated)
        stats.append({"frame": f_idx, "good": n_good, "inliers": n_inlier,
                       "detected": detected})

    return processed, stats


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 4: Object Recognition Using Video (SIFT)")
    print("=" * 60)

    ref_img = create_reference_object()
    frames  = create_synthetic_video(ref_img, n_frames=30)

    print(f"\nProcessing {len(frames)} video frames …")
    processed, stats = process_video_frames(ref_img, frames, min_good=6)

    n_detected = sum(s["detected"] for s in stats)
    print(f"\nDetected in {n_detected}/{len(frames)} frames")

    # ── Save annotated video ──────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task4_sift_video_output.avi"
    h, w     = processed[0].shape[:2]
    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    writer   = cv2.VideoWriter(out_path, fourcc, 10, (w, h))
    for f in processed:
        writer.write(f)
    writer.release()
    print(f"Video saved → {out_path}")

    # ── Static figure: sample frames ──────────────────────────────────────────
    sample_indices = np.linspace(0, len(processed) - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Task 4 – SIFT Object Recognition in Video Frames",
                 fontsize=14, fontweight="bold")

    for ax, idx in zip(axes.flat, sample_indices):
        ax.imshow(cv2.cvtColor(processed[idx], cv2.COLOR_BGR2RGB))
        s = stats[idx]
        title = (f"Frame {idx+1} | {'✔ Detected' if s['detected'] else '✘ Not found'} "
                 f"| m={s['good']}")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/task4_sift_video_frames.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Detection stats chart ─────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 4))
    frame_ids = [s["frame"] + 1 for s in stats]
    good_vals = [s["good"]     for s in stats]
    colors    = ["green" if s["detected"] else "red" for s in stats]
    ax.bar(frame_ids, good_vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(6, color="orange", linestyle="--", label="Min threshold (6)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Good SIFT matches")
    ax.set_title("SIFT Match Count per Frame (green = detected)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/task4_match_stats.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figures saved → outputs/task4_sift_video_frames.png, outputs/task4_match_stats.png")


if __name__ == "__main__":
    main()

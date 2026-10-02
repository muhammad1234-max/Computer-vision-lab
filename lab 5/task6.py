"""
task6.py
Task 6 - Segmenting a Region Inside a Medical Image (Region Growing)

Image: brain.png (brain MRI -- regions of similar intensity).

Algorithm (implemented from scratch -- no prebuilt segmentation library):
  1. Select a seed pixel.
  2. Push seed onto a queue.
  3. For each pixel in the queue, if |pixel_intensity - seed_intensity| <= threshold:
       - Label it as part of the region (mask = 255).
       - Push its unvisited 8-connected neighbours onto the queue.
  4. Continue until queue is empty.

Parameter study:
  - 3 different intensity thresholds : 5, 15, 25
  - 2 different seed locations

Why seed location matters -- explained in console output.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file without display
import matplotlib.pyplot as plt
from collections import deque
import os

os.makedirs("outputs", exist_ok=True)

# -- Load image ----------------------------------------------------------------
img = cv2.imread("images/brain.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    # Try JPEG fallback (Wikipedia brain MRI may be saved as .jpg)
    img = cv2.imread("images/brain.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    # Generate a synthetic brain-like image for demonstration
    print("WARNING: brain.png not found -- generating synthetic brain MRI.")
    h, w = 400, 400
    img = np.zeros((h, w), dtype=np.uint8)

    # Skull (bright ring)
    cv2.ellipse(img, (200, 200), (180, 160), 0, 0, 360, 220, -1)
    cv2.ellipse(img, (200, 200), (165, 145), 0, 0, 360,   0, -1)

    # White matter (bright region)
    cv2.ellipse(img, (200, 200), (145, 125), 0, 0, 360, 180, -1)

    # Gray matter (darker ring)
    cv2.ellipse(img, (200, 200), (120, 100), 0, 0, 360, 120, -1)

    # Ventricles (dark CSF-filled)
    cv2.ellipse(img, (190, 195), (25, 35),  0, 0, 360,  40, -1)
    cv2.ellipse(img, (210, 195), (25, 35),  0, 0, 360,  40, -1)

    # Add Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img   = cv2.GaussianBlur(img, (5, 5), 2)

    cv2.imwrite("images/brain.png", img)
    print("  Synthetic brain MRI saved to images/brain.png")

print(f"Brain image shape: {img.shape}  |  Intensity range: {img.min()}-{img.max()}")

# -- Region-growing algorithm (from scratch) ------------------------------------
NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def region_grow(gray, seed_y, seed_x, threshold):
    """
    Grow a region from (seed_y, seed_x) using 8-connectivity.
    A neighbour is added to the region if its intensity differs from the
    seed pixel's intensity by at most `threshold`.

    Returns a binary mask (0 or 255) of the same size as gray.
    """
    rows, cols = gray.shape
    seed_val   = int(gray[seed_y, seed_x])

    visited = np.zeros((rows, cols), dtype=bool)
    mask    = np.zeros((rows, cols), dtype=np.uint8)

    queue            = deque()
    queue.append((seed_y, seed_x))
    visited[seed_y, seed_x] = True

    while queue:
        y, x = queue.popleft()
        if abs(int(gray[y, x]) - seed_val) <= threshold:
            mask[y, x] = 255
            for dy, dx in NEIGHBORS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return mask

# -- Define seeds and thresholds -----------------------------------------------
h, w = img.shape

# Seed A -- bright skull/white-matter ring region (row=80, col=200 -> intensity~180)
seed_A = (80,  w // 2)
# Seed B -- darker gray-matter / inner region (row=160, col=200 -> intensity~107)
seed_B = (int(h * 0.4), w // 2)

thresholds = [5, 15, 25]

seeds = [
    (seed_A, "Seed A (Bright Region ~180)"),
    (seed_B, "Seed B (Gray Region ~107)"),
]

print(f"\nSeed A: {seed_A}  intensity={img[seed_A[0], seed_A[1]]}")
print(f"Seed B: {seed_B}  intensity={img[seed_B[0], seed_B[1]]}")

# -- Compute all masks ---------------------------------------------------------
results = {}   # (seed_label, thresh) -> mask
for (seed_yx, seed_label) in seeds:
    for t in thresholds:
        mask = region_grow(img, seed_yx[0], seed_yx[1], t)
        results[(seed_label, t)] = mask
        n = np.count_nonzero(mask)
        print(f"  {seed_label}, T={t:2d} -> {n:,} pixels ({100*n/img.size:.1f}%)")

# -- Output figure -------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.patch.set_facecolor("#0d1117")

seed_colours = {
    "Seed A (Bright Region ~180)": "#ff6b6b",
    "Seed B (Gray Region ~107)":   "#4fc3f7",
}
thresh_colours = ["#81c995", "#ffd700", "#ff9800"]

for row_idx, ((seed_yx, seed_label)) in enumerate(seeds):
    # Column 0: original with seed marked
    orig_marked = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    sy, sx = seed_yx
    cv2.circle(orig_marked, (sx, sy), 8,
               (255, 80, 80) if row_idx == 0 else (80, 180, 255), -1)
    cv2.circle(orig_marked, (sx, sy), 10,
               (255, 255, 255), 2)

    axes[row_idx, 0].imshow(orig_marked)
    axes[row_idx, 0].set_title(
        f"Original\n{seed_label}\n@ pixel {seed_yx}",
        color=seed_colours[seed_label], fontsize=9, fontweight="bold",
    )
    axes[row_idx, 0].axis("off")

    for col_idx, t in enumerate(thresholds, start=1):
        mask = results[(seed_label, t)]
        # Overlay mask in colour on original
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).copy()
        overlay[mask == 255] = [80, 200, 120] if row_idx == 0 else [80, 150, 255]

        axes[row_idx, col_idx].imshow(overlay)
        n = np.count_nonzero(mask)
        axes[row_idx, col_idx].set_title(
            f"T = {t}\n{n:,} pixels  ({100*n/img.size:.1f}%)",
            color=thresh_colours[col_idx - 1], fontsize=10, fontweight="bold",
        )
        axes[row_idx, col_idx].axis("off")

fig.suptitle(
    "Task 6 - Region Growing on Brain MRI\n"
    "Row 1: Seed A (White Matter)  |  Row 2: Seed B (Gray Matter)\n"
    "Columns: Original with seed  ->  T=5  ->  T=15  ->  T=25",
    color="white", fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("outputs/task6_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 6 -- REGION GROWING ANALYSIS")
print("=" * 65)
print()
print("ALGORITHM OVERVIEW:")
print("  Starting from a seed pixel, the algorithm uses a BFS queue.")
print("  Each dequeued pixel is added to the mask if:")
print("    |pixel_intensity - seed_intensity| <= threshold")
print("  Its 8-connected unvisited neighbours are then enqueued.")
print("  The process stops when the queue is empty.")
print()
print("EFFECT OF THRESHOLD:")
print("  T =  5  Very conservative. Only pixels nearly identical to the seed.")
print("           Small, tightly bounded region. Useful for isolating a")
print("           very specific tissue type without leaking.")
print("  T = 15  Moderate. Captures the main tissue region including mild")
print("           intensity variation. Good trade-off for brain structures.")
print("  T = 25  Liberal. The region grows into neighbouring tissues that")
print("           have similar but not identical intensity. Risk of leaking")
print("           across tissue boundaries (e.g., white -> grey matter).")
print()
print("WHY CHANGING THE SEED POINT SIGNIFICANTLY CHANGES THE RESULT:")
print("-" * 65)
print("  The region-growing algorithm compares every candidate pixel to the")
print("  seed's intensity. Different anatomical regions have different mean")
print("  intensities (in brain MRI: white matter > grey matter > CSF).")
print()
print("  - Seed A placed in white matter (high intensity ~180):")
print("    The threshold window [180-T, 180+T] covers other white-matter")
print("    voxels, growing a large connected white-matter region.")
print()
print("  - Seed B placed in grey matter (mid intensity ~120):")
print("    The window [120-T, 120+T] matches grey-matter voxels.")
print("    White matter pixels (~180) lie far outside this window, so the")
print("    growth stops at the white/grey matter boundary.")
print()
print("  A single image therefore yields completely different segmented")
print("  regions depending on the seed, because the seed determines which")
print("  intensity range the algorithm is allowed to grow into. This is")
print("  why domain knowledge (knowing which tissue to select) is essential.")
print("=" * 65)
print("\nOutput saved -> outputs/task6_output.png")

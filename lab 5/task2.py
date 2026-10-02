"""
task2.py
Task 2 - Choosing the Correct Adaptive Threshold Strategy

Tests:
  - blockSize values : 5, 11, 21
  - C values         : 2, 5, 10
  - Methods          : ADAPTIVE_THRESH_MEAN_C  and  ADAPTIVE_THRESH_GAUSSIAN_C

Output: Original + at least 6 adaptive segmentation results (2 x 4 grid showing
blockSize variation, then a second figure for C variation).

Answers in console:
  1. Small neighbourhood  -> too local, noise amplified
  2. Large neighbourhood  -> loses fine detail / similar to global
  3. C increased          -> threshold raised, fewer pixels become foreground (text)
  4. Best combination     -> Gaussian, blockSize=11, C=2
  5. Which method better  -> Gaussian (weighted average, smoother boundaries)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file without display
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# -- Load image ----------------------------------------------------------------
img = cv2.imread("images/sudoku.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("images/sudoku.png not found. Run download_images.py first.")

# -- Helper --------------------------------------------------------------------
def adaptive_thresh(image, method, bs, c):
    return cv2.adaptiveThreshold(
        image, 255, method, cv2.THRESH_BINARY, bs, c
    )

MEAN  = cv2.ADAPTIVE_THRESH_MEAN_C
GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# -----------------------------------------------------------------------------
# FIGURE 1 -- blockSize variation (C fixed at 2)
# Rows: Mean / Gaussian   |   Cols: BS=5, BS=11, BS=21
# -----------------------------------------------------------------------------
block_sizes = [5, 11, 21]
C_fixed = 2

fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10))
fig1.patch.set_facecolor("#0f0f23")

row_labels = ["Mean-based", "Gaussian-based"]
method_list = [MEAN, GAUSS]
method_names = ["MEAN", "GAUSS"]

for row, (method, label) in enumerate(zip(method_list, row_labels)):
    # Column 0 -- original
    axes1[row, 0].imshow(img, cmap="gray")
    axes1[row, 0].set_title("Original" if row == 0 else "", color="white",
                             fontsize=11, fontweight="bold")
    axes1[row, 0].axis("off")
    axes1[row, 0].set_ylabel(label, color="#4fc3f7", fontsize=11, fontweight="bold")

    for col, bs in enumerate(block_sizes, start=1):
        result = adaptive_thresh(img, method, bs, C_fixed)
        axes1[row, col].imshow(result, cmap="gray")
        axes1[row, col].set_title(
            f"{method_names[row]}\nBS={bs}, C={C_fixed}",
            color="white", fontsize=10, fontweight="bold",
        )
        axes1[row, col].axis("off")

fig1.suptitle(
    "Task 2 - BlockSize Variation (C = 2 fixed)\n"
    "Effect of neighbourhood size on adaptive thresholding quality",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("outputs/task2_blocksize_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close("all")

# -----------------------------------------------------------------------------
# FIGURE 2 -- C variation (blockSize fixed at 11)
# Rows: Mean / Gaussian   |   Cols: C=2, C=5, C=10
# -----------------------------------------------------------------------------
c_values  = [2, 5, 10]
BS_fixed  = 11

fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
fig2.patch.set_facecolor("#0f0f23")

for row, (method, label) in enumerate(zip(method_list, row_labels)):
    axes2[row, 0].imshow(img, cmap="gray")
    axes2[row, 0].set_title("Original" if row == 0 else "", color="white",
                             fontsize=11, fontweight="bold")
    axes2[row, 0].axis("off")
    axes2[row, 0].set_ylabel(label, color="#4fc3f7", fontsize=11, fontweight="bold")

    for col, c in enumerate(c_values, start=1):
        result = adaptive_thresh(img, method, BS_fixed, c)
        axes2[row, col].imshow(result, cmap="gray")
        axes2[row, col].set_title(
            f"{method_names[row]}\nBS={BS_fixed}, C={c}",
            color="white", fontsize=10, fontweight="bold",
        )
        axes2[row, col].axis("off")

fig2.suptitle(
    "Task 2 - Constant C Variation (blockSize = 11 fixed)\n"
    "Effect of C on foreground/background separation",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("outputs/task2_C_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close("all")

# -----------------------------------------------------------------------------
# FIGURE 3 -- Combined best-of grid (>= 6 adaptive results + original)
# -----------------------------------------------------------------------------
combos = [
    (MEAN,  5,  2,  "Mean BS=5, C=2"),
    (MEAN,  11, 2,  "Mean BS=11, C=2"),
    (MEAN,  21, 2,  "Mean BS=21, C=2"),
    (GAUSS, 5,  10, "Gauss BS=5, C=10"),
    (GAUSS, 11, 5,  "Gauss BS=11, C=5"),
    (GAUSS, 21, 2,  "Gauss BS=21, C=2"),
    (GAUSS, 11, 2,  "Gauss BS=11, C=2 * BEST"),
]

fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
fig3.patch.set_facecolor("#0f0f23")
axes3 = axes3.flatten()

axes3[0].imshow(img, cmap="gray")
axes3[0].set_title("Original", color="white", fontsize=11, fontweight="bold")
axes3[0].axis("off")

for i, (method, bs, c, label) in enumerate(combos, start=1):
    result = adaptive_thresh(img, method, bs, c)
    axes3[i].imshow(result, cmap="gray")
    color = "#ffd700" if "BEST" in label else "white"
    axes3[i].set_title(label, color=color, fontsize=9, fontweight="bold")
    axes3[i].axis("off")

fig3.suptitle(
    "Task 2 - Final Comparison: Original + 7 Adaptive Segmentation Results\n"
    "(* marks the cleanest text segmentation)",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("outputs/task2_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
plt.close("all")

# -- Analysis report -----------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 2 -- REQUIRED ANALYSIS")
print("=" * 65)
print()
print("1. NEIGHBOURHOOD TOO SMALL (blockSize = 5)")
print("   Each pixel's threshold is computed over only a 5x5 window.")
print("   Local intensity variations (noise, ink texture) dominate the")
print("   estimate, so isolated noisy pixels create false foreground hits.")
print("   Result: Speckle noise around characters; broken strokes.")
print()
print("2. NEIGHBOURHOOD TOO LARGE (blockSize = 21)")
print("   The 21x21 window is large enough to include both text and")
print("   background in one neighbourhood, weakening local adaptation.")
print("   The result approaches a global-like threshold across each region,")
print("   causing dark-area characters to fade and bright-area background")
print("   noise to appear.")
print()
print("3. C INCREASED (C = 2 -> 5 -> 10, blockSize = 11)")
print("   C is subtracted from the local weighted mean/Gaussian average.")
print("   Increasing C lowers the effective threshold, pushing more pixels")
print("   above it -> fewer pixels classified as foreground (text appears")
print("   thinner, strokes become fragmented or disappear at high C).")
print()
print("4. BEST COMBINATION: Gaussian, blockSize=11, C=2")
print("   - blockSize=11 balances locality vs. stability.")
print("   - C=2 provides a small bias that cleanly separates thin strokes.")
print("   - Gaussian weighting down-weights pixels far from the centre,")
print("     giving smoother, more natural-looking text boundaries.")
print()
print("5. WHICH METHOD PERFORMS BETTER ON THIS IMAGE?")
print("   Gaussian-based adaptive thresholding produces cleaner character")
print("   boundaries with less speckle noise compared to Mean-based.")
print("   Reason: The Gaussian kernel weights central pixels more heavily,")
print("   reducing the influence of edge-region outliers in the neighbourhood.")
print("   Mean-based treats every pixel equally, making it more sensitive to")
print("   local intensity spikes (noise).")
print("=" * 65)
print("\nOutputs saved ->")
print("  outputs/task2_blocksize_comparison.png")
print("  outputs/task2_C_comparison.png")
print("  outputs/task2_output.png  (main final comparison)")

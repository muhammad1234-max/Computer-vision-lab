"""
task3.py
Task 3 - Quality-Control System Using Otsu's Thresholding

Image: coins.png (OpenCV watershed example -- objects on consistent background).

Steps:
  1. Convert to grayscale.
  2. Generate histogram.
  3. Apply Otsu's thresholding -> record the automatically chosen threshold.
  4. Repeat after applying slight contrast change (CLAHE).
  5. Compare automatically selected thresholds.

Required output: Original -> Histogram -> Otsu Binary Mask

Why Otsu is useful: explained in the console report.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("outputs", exist_ok=True)

# -- 1. Load and convert to grayscale -----------------------------------------
img_color = cv2.imread("images/coins.png")
if img_color is None:
    raise FileNotFoundError("images/coins.png not found. Run download_images.py first.")

img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"Image shape: {img.shape}  |  Intensity range: {img.min()}-{img.max()}")

# -- 2. Otsu's thresholding (original image) -----------------------------------
otsu_thresh_val, otsu_mask = cv2.threshold(
    img, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU   # THRESH_BINARY_INV so coins = white
)
print(f"\nOtsu threshold (original): {otsu_thresh_val:.1f}")

# -- 3. Histogram data ---------------------------------------------------------
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

# -- 4. Contrast change -- apply CLAHE to modify local contrast -----------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

otsu_thresh_clahe, otsu_mask_clahe = cv2.threshold(
    img_clahe, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
print(f"Otsu threshold (CLAHE-enhanced): {otsu_thresh_clahe:.1f}")

hist_clahe = cv2.calcHist([img_clahe], [0], None, [256], [0, 256]).flatten()

# -- 5. Build output figure: Original -> Histogram -> Otsu Binary Mask -----------
fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)

# ---------- Row 1: Original image ----------
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.imshow(img, cmap="gray")
ax_orig.set_title("Original\n(Grayscale)", color="white", fontsize=12, fontweight="bold")
ax_orig.axis("off")

# ---------- Row 1: Histogram with Otsu line ----------
ax_hist = fig.add_subplot(gs[0, 1])
ax_hist.set_facecolor("#161b22")
ax_hist.bar(range(256), hist, color="#4fc3f7", alpha=0.8, width=1)
ax_hist.axvline(otsu_thresh_val, color="#ff6b6b", linewidth=2.5,
                label=f"Otsu T = {int(otsu_thresh_val)}")
ax_hist.set_xlim(0, 255)
ax_hist.set_xlabel("Pixel Intensity", color="white")
ax_hist.set_ylabel("Frequency", color="white")
ax_hist.set_title("Histogram\n(Original)", color="white", fontsize=12, fontweight="bold")
ax_hist.tick_params(colors="white")
ax_hist.spines[:].set_color("#30363d")
ax_hist.legend(facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", fontsize=10)

# ---------- Row 1: Otsu mask ----------
ax_otsu = fig.add_subplot(gs[0, 2])
ax_otsu.imshow(otsu_mask, cmap="gray")
ax_otsu.set_title(f"Otsu Binary Mask\n(T = {int(otsu_thresh_val)})",
                  color="white", fontsize=12, fontweight="bold")
ax_otsu.axis("off")

# ---------- Row 2: CLAHE-enhanced ----------
ax_clahe_img = fig.add_subplot(gs[1, 0])
ax_clahe_img.imshow(img_clahe, cmap="gray")
ax_clahe_img.set_title("CLAHE-Enhanced\n(Contrast Changed)", color="white",
                        fontsize=12, fontweight="bold")
ax_clahe_img.axis("off")

ax_hist2 = fig.add_subplot(gs[1, 1])
ax_hist2.set_facecolor("#161b22")
ax_hist2.bar(range(256), hist_clahe, color="#81c995", alpha=0.8, width=1)
ax_hist2.axvline(otsu_thresh_clahe, color="#ff6b6b", linewidth=2.5,
                 label=f"Otsu T = {int(otsu_thresh_clahe)}")
ax_hist2.set_xlim(0, 255)
ax_hist2.set_xlabel("Pixel Intensity", color="white")
ax_hist2.set_ylabel("Frequency", color="white")
ax_hist2.set_title("Histogram\n(CLAHE-Enhanced)", color="white",
                   fontsize=12, fontweight="bold")
ax_hist2.tick_params(colors="white")
ax_hist2.spines[:].set_color("#30363d")
ax_hist2.legend(facecolor="#21262d", edgecolor="#30363d",
                labelcolor="white", fontsize=10)

ax_otsu2 = fig.add_subplot(gs[1, 2])
ax_otsu2.imshow(otsu_mask_clahe, cmap="gray")
ax_otsu2.set_title(f"Otsu Binary Mask - CLAHE\n(T = {int(otsu_thresh_clahe)})",
                   color="white", fontsize=12, fontweight="bold")
ax_otsu2.axis("off")

fig.suptitle(
    "Task 3 - Otsu's Thresholding Quality-Control System\n"
    "Original  ->  Histogram  ->  Otsu Binary Mask",
    color="white", fontsize=14, fontweight="bold", y=1.01,
)
plt.savefig("outputs/task3_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Console report ------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 3 -- OTSU'S THRESHOLDING REPORT")
print("=" * 65)
print(f"  Automatically selected threshold (original) : {int(otsu_thresh_val)}")
print(f"  Automatically selected threshold (CLAHE)    : {int(otsu_thresh_clahe)}")
print()
print("WHY DOES THE THRESHOLD CHANGE AFTER CONTRAST ADJUSTMENT?")
print("-" * 65)
print("  CLAHE redistributes intensities across the full [0,255] range.")
print("  The histogram valley between the two peaks (coins vs. background)")
print("  shifts position, so Otsu's algorithm finds a different optimal split.")
print()
print("WHY IS OTSU USEFUL (vs. manual thresholding)?")
print("-" * 65)
print("  In a quality-control system, images arrive from varying lighting")
print("  conditions and camera exposures. Manually choosing a threshold T")
print("  each time is impractical and inconsistent.")
print()
print("  Otsu's method analyses the histogram and automatically finds the")
print("  threshold that MAXIMISES inter-class variance (the separation")
print("  between the two intensity peaks: object pixels and background pixels).")
print()
print("  For an image with a bimodal histogram -- like coins on a uniform")
print("  background -- Otsu gives the optimal binarisation without any human")
print("  parameter tuning. The engineer can run the same algorithm on every")
print("  image and always get a suitable, data-driven threshold.")
print("=" * 65)
print("\nOutput saved -> outputs/task3_output.png")

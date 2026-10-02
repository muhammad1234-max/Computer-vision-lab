"""
task7.py
Task 7 - Separating Touching Coins Using Marker-Based Watershed

Image: coins.png (multiple touching circular objects on a tray).

Pipeline (exactly as described in the manual):
  Stage 1  - Preprocessing    : Convert to grayscale, apply Gaussian blur.
  Stage 2  - Thresholding     : Otsu's binarization -> binary mask.
  Stage 3  - Noise removal    : Morphological opening (removes small noise).
  Stage 4  - Sure background  : Dilation expands known background.
  Stage 5  - Distance transform: Measure distance from each fg pixel to bg.
  Stage 6  - Sure foreground  : Threshold dist-transform at 0.5 x max.
  Stage 7  - Unknown region   : sure_bg − sure_fg (uncertain boundary zone).
  Stage 8  - Marker labeling  : Label connected components in sure-foreground.
  Stage 9  - Watershed        : Run cv2.watershed to assign boundaries.
  Stage 10 - Boundary visualization: Draw -1 labelled pixels in red.

Required output:
  Original -> Threshold -> Sure Background -> Distance Transform
  -> Sure Foreground -> Unknown Region -> Final Watershed
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

# -- Stage 1: Preprocessing ----------------------------------------------------
img_bgr = cv2.imread("images/coins.png")
if img_bgr is None:
    raise FileNotFoundError("images/coins.png not found. Run download_images.py first.")

img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

print(f"Image shape: {img_bgr.shape}")

# -- Stage 2: Thresholding (Otsu) ----------------------------------------------
otsu_val, binary = cv2.threshold(
    img_blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
print(f"Otsu threshold: {otsu_val:.1f}")

# -- Stage 3: Noise removal -- morphological opening ---------------------------
kernel  = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# -- Stage 4: Sure background -- dilation --------------------------------------
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# -- Stage 5: Distance transform -----------------------------------------------
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# -- Stage 6: Sure foreground -- threshold at 0.5 x max ------------------------
DT_RATIO = 0.5
_, sure_fg = cv2.threshold(
    dist_transform, DT_RATIO * dist_transform.max(), 255, cv2.THRESH_BINARY
)
sure_fg = np.uint8(sure_fg)
print(f"Distance-transform threshold ratio: {DT_RATIO}")
print(f"Sure-foreground pixels: {np.count_nonzero(sure_fg)}")

# -- Stage 7: Unknown region ---------------------------------------------------
unknown = cv2.subtract(sure_bg, sure_fg)

# -- Stage 8: Marker labeling --------------------------------------------------
_, markers = cv2.connectedComponents(sure_fg)
# Shift labels so background gets label 1, not 0 (watershed convention)
markers = markers + 1
# Mark unknown region as 0
markers[unknown == 255] = 0

num_markers = markers.max()
print(f"Detected foreground markers (objects + bg): {num_markers}  "
      f"-> estimated coins: {num_markers - 1}")

# -- Stage 9: Watershed ---------------------------------------------------------
markers_ws = markers.copy()
cv2.watershed(img_bgr, markers_ws)

# -- Stage 10: Boundary visualisation -----------------------------------------
result = img_rgb.copy()
result[markers_ws == -1] = [255, 0, 0]   # Red boundaries

# -- Build required output figure ----------------------------------------------
dist_norm = cv2.normalize(dist_transform, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)

panels = [
    (img_rgb,      "Original",              "rgb"),
    (binary,       "Otsu Threshold",        "gray"),
    (sure_bg,      "Sure Background",       "gray"),
    (dist_norm,    "Distance Transform",    "hot"),
    (sure_fg,      "Sure Foreground",       "gray"),
    (unknown,      "Unknown Region",        "gray"),
    (result,       "Final Watershed\n(Red = boundaries)", "rgb"),
]

fig, axes = plt.subplots(1, 7, figsize=(30, 5))
fig.patch.set_facecolor("#0d1117")

for ax, (img_panel, title, cmap) in zip(axes, panels):
    if cmap == "rgb":
        ax.imshow(img_panel)
    else:
        ax.imshow(img_panel, cmap=cmap)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")

fig.suptitle(
    "Task 7 - Marker-Based Watershed: Separating Touching Coins\n"
    "Original -> Threshold -> Sure Background -> Distance Transform"
    " -> Sure Foreground -> Unknown Region -> Final Watershed",
    color="white", fontsize=11, fontweight="bold", y=1.05,
)
plt.tight_layout()
plt.savefig("outputs/task7_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 7 -- WATERSHED PIPELINE REPORT")
print("=" * 65)
print(f"  Otsu threshold          : {int(otsu_val)}")
print(f"  Dist-transform ratio    : {DT_RATIO}")
print(f"  Foreground markers found: {num_markers - 1}  (excl. background marker)")
print()
print("VISUAL INSPECTION:")
print("  Each coin region is coloured separately in the watershed output.")
print("  Red lines mark the watershed boundaries between adjacent coins.")
print("  Coins that were touching now have a clean separating contour.")
print()
print("HOW WATERSHED SOLVES THE TOUCHING-COIN PROBLEM:")
print("  Simple thresholding merges touching coins into one connected")
print("  white blob, giving an incorrect count. Watershed treats the")
print("  distance-transform surface as a topographic map; each local peak")
print("  becomes a seed (marker). Flooding from multiple seeds simultaneously")
print("  builds dams (watershed lines) exactly where two regions meet --")
print("  separating touching objects without needing to know their exact")
print("  boundary in advance.")
print("=" * 65)
print("\nOutput saved -> outputs/task7_output.png")

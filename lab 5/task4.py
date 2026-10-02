"""
task4.py
Task 4 - Sorting Objects by Color (HSV-based Segmentation)

Image: smarties.png (multi-colour candy objects on a background).

Strategy: Isolate YELLOW smarties using HSV thresholding.
  - HSV is preferred over BGR because Hue separates colour identity
    from brightness, making colour-based segmentation lighting-robust.

Two masks produced:
  - Restrictive mask  : very narrow hue band  (H 22-28) -- misses many yellow pixels
  - Complete mask     : wider hue band        (H 15-40) -- captures full object

Steps:
  1. Load in colour.
  2. Convert to HSV.
  3. Define lower/upper bounds.
  4. Generate masks.
  5. Extract the selected colour from the original using the mask.
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

# -- 1. Load image in colour ---------------------------------------------------
img_bgr = cv2.imread("images/smarties.png")
if img_bgr is None:
    raise FileNotFoundError("images/smarties.png not found. Run download_images.py first.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # for matplotlib display

# -- 2. Convert to HSV ---------------------------------------------------------
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
print(f"Image shape: {img_bgr.shape}")
print("HSV ranges -- H: [0,179]  S: [0,255]  V: [0,255]")

# -- 3 & 4. Define bounds and generate masks -----------------------------------

# ---- RESTRICTIVE mask -- narrow yellow band ----
lower_restrict = np.array([22,  120, 80],  dtype=np.uint8)
upper_restrict = np.array([28,  255, 255], dtype=np.uint8)
mask_restrict  = cv2.inRange(img_hsv, lower_restrict, upper_restrict)

# ---- COMPLETE mask -- wider yellow band ----
lower_complete = np.array([15,  80,  60],  dtype=np.uint8)
upper_complete = np.array([40,  255, 255], dtype=np.uint8)
mask_complete  = cv2.inRange(img_hsv, lower_complete, upper_complete)

# Optional: clean the complete mask with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_complete_clean = cv2.morphologyEx(mask_complete, cv2.MORPH_OPEN,  kernel)
mask_complete_clean = cv2.morphologyEx(mask_complete_clean, cv2.MORPH_CLOSE, kernel)

# -- 5. Extract colour from original using masks -------------------------------
extracted_restrict = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_restrict)
extracted_complete = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_complete_clean)

# -- 6. Build output figure ----------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.patch.set_facecolor("#0d1117")

rows_data = [
    # (title_row, image, mask, extraction)
    ("RESTRICTIVE Mask\nH=[22,28] S=[120,255]",
     img_rgb, mask_restrict, extracted_restrict),
    ("COMPLETE Mask\nH=[15,40] S=[80,255]",
     img_rgb, mask_complete_clean, extracted_complete),
]

col_titles = ["Original (RGB)", "HSV Hue Channel", "Binary Mask", "Extracted Yellow"]
for col_idx, col_title in enumerate(col_titles):
    axes[0, col_idx].set_title(col_title, color="white", fontsize=11, fontweight="bold")

for row_idx, (row_label, orig, mask, extracted) in enumerate(rows_data):
    # Col 0: Original
    axes[row_idx, 0].imshow(orig)
    axes[row_idx, 0].set_ylabel(row_label, color="#ffd700" if row_idx == 0 else "#4fc3f7",
                                fontsize=10, fontweight="bold")
    axes[row_idx, 0].axis("off")

    # Col 1: Hue channel
    hue = img_hsv[:, :, 0]
    axes[row_idx, 1].imshow(hue, cmap="hsv")
    axes[row_idx, 1].axis("off")

    # Col 2: Binary mask
    axes[row_idx, 2].imshow(mask, cmap="gray")
    axes[row_idx, 2].axis("off")

    # Col 3: Extracted colour
    axes[row_idx, 3].imshow(extracted)
    axes[row_idx, 3].axis("off")

fig.suptitle(
    "Task 4 - HSV Colour-Based Segmentation  (Target: Yellow Smarties)\n"
    "Row 1: Restrictive Mask  |  Row 2: Complete Mask",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("outputs/task4_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 4 -- HSV COLOUR SEGMENTATION REPORT")
print("=" * 65)
print()
print("Target colour: YELLOW smarties")
print()
print("RESTRICTIVE MASK  ->  lower=(22,120,80)  upper=(28,255,255)")
print("  Hue range [22,28]: extremely narrow -- only saturated, mid-tone yellow.")
print("  Saturation >= 120 : excludes slightly desaturated / washed-out pixels.")
print("  Result: Many yellow candy pixels are MISSED (darker edges, lighter")
print("  highlights of the candy surface fall outside this tight range).")
print("  Why it fails: Natural objects are not a single HSV point;")
print("  curvature, specular highlights, and shadows all shift H, S, V.")
print()
print("COMPLETE MASK  ->  lower=(15,80,60)  upper=(40,255,255)")
print("  Hue range [15,40]: covers golden-yellow through lemon-yellow,")
print("  accommodating the full shade variation across the candy surface.")
print("  Saturation >= 80  : still excludes near-grey / white pixels.")
print("  Value >= 60       : excludes very dark shadow pixels.")
print("  Morphological open+close applied to remove speckle and fill holes.")
print("  Result: Yellow candy region captured more completely.")
print()
print("WHY RESTRICTIVE MASK FAILS:")
print("  The candy surface is curved, so illumination varies across it.")
print("  Specular highlights push V high and S low. Shadow regions push V low.")
print("  A narrow [H_min, H_max] range cannot span the full appearance of")
print("  one object under realistic illumination. Widening the HSV range while")
print("  keeping saturation/value floors prevents false positives from")
print("  near-white or near-black pixels.")
print()
print("WHY HSV IS PREFERRED OVER BGR:")
print("  BGR represents colour as a mixture of R, G, B -- intensity-dependent.")
print("  Illumination changes affect all three channels simultaneously.")
print("  HSV separates Hue (colour identity) from Saturation (richness) and")
print("  Value (brightness). A yellow object under dim light still has a")
print("  similar Hue; only V changes. This separation makes HSV thresholding")
print("  far more robust to lighting variation than BGR thresholding.")
print("=" * 65)
print("\nOutput saved -> outputs/task4_output.png")

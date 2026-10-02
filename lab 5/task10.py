"""
task10.py
Task 10 - Design a Segmentation System for an Unknown Image

Image chosen: smarties.png
Reason: It contains multiple coloured circular objects on a background --
a challenging scene involving colour variety, touching objects, and
structured boundaries, making it suitable for multiple methods.

Three segmentation methods applied:
  Method 1 - Otsu's Global Thresholding
    Assumption: bimodal intensity histogram (objects vs. background).
  Method 2 - HSV Colour-Based Thresholding
    Assumption: target region has a distinct, isolatable hue.
  Method 3 - Canny Edge Detection
    Assumption: boundaries between regions are marked by sharp intensity changes.

For each method:
  - What assumption the method makes about the image
  - What region/object it successfully identifies
  - What part of the image it incorrectly segments
  - Which parameter has the greatest effect on the result

Final comparison: Original | Method 1 | Method 2 | Method 3
Technical conclusion: which method produces the most meaningful regions.
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
img_bgr = cv2.imread("images/smarties.png")
if img_bgr is None:
    raise FileNotFoundError("images/smarties.png not found. Run download_images.py first.")

img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

print(f"Image: smarties.png  |  Shape: {img_bgr.shape}")

# -----------------------------------------------------------------------------
# METHOD 1 -- Otsu's Global Thresholding
# Assumption: there are two dominant intensity classes (bright candies vs dark bg)
# -----------------------------------------------------------------------------
otsu_val, otsu_mask = cv2.threshold(
    img_blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
# Convert single-channel mask to 3-ch for overlay
otsu_overlay = img_rgb.copy()
otsu_overlay[otsu_mask == 0] = [30, 30, 30]  # darken background

print(f"Method 1 -- Otsu threshold: {int(otsu_val)}")

# -----------------------------------------------------------------------------
# METHOD 2 -- HSV Colour-Based Thresholding  (segment RED smarties)
# Assumption: Red objects have a distinct hue separable from other colours
# Red wraps around Hue 0/179 in OpenCV -- use two ranges
# -----------------------------------------------------------------------------
lower_red1 = np.array([0,   100, 80],  dtype=np.uint8)
upper_red1 = np.array([10,  255, 255], dtype=np.uint8)
lower_red2 = np.array([165, 100, 80],  dtype=np.uint8)
upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

mask_r1   = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask_r2   = cv2.inRange(img_hsv, lower_red2, upper_red2)
hsv_mask  = cv2.bitwise_or(mask_r1, mask_r2)

# Morphological cleanup
kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN,  kernel)
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

hsv_overlay = img_rgb.copy()
# Non-red areas darkened
hsv_overlay[hsv_mask == 0] = [30, 30, 30]
red_pixel_count = np.count_nonzero(hsv_mask)
print(f"Method 2 -- Red pixels detected: {red_pixel_count:,}")

# -----------------------------------------------------------------------------
# METHOD 3 -- Canny Edge Detection
# Assumption: boundaries between objects and background are sharp
# -----------------------------------------------------------------------------
CANNY_LOW  = 50
CANNY_HIGH = 150
edges = cv2.Canny(img_blur, CANNY_LOW, CANNY_HIGH)

# Convert edges to coloured overlay on original
canny_overlay = img_rgb.copy()
canny_overlay[edges > 0] = [255, 80, 80]   # Red highlights on boundaries

edge_count = np.count_nonzero(edges)
print(f"Method 3 -- Edge pixels detected: {edge_count:,}  "
      f"(low={CANNY_LOW}, high={CANNY_HIGH})")

# -- Final Comparison Figure: Original | Method 1 | Method 2 | Method 3 -------
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.patch.set_facecolor("#0d1117")

panels = [
    (img_rgb,       "Original\n(smarties.png)",                      None),
    (otsu_overlay,  f"Method 1\nOtsu T={int(otsu_val)}\n(Global Threshold)", "#ffd700"),
    (hsv_overlay,   "Method 2\nHSV Red Detection\n(Colour Threshold)",       "#ff6b6b"),
    (canny_overlay, f"Method 3\nCanny Edges\n(low={CANNY_LOW}, high={CANNY_HIGH})", "#4fc3f7"),
]

for ax, (img_panel, title, col) in zip(axes, panels):
    ax.imshow(img_panel)
    tc = col if col else "white"
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=8)
    ax.axis("off")

fig.suptitle(
    "Task 10 - Segmentation System Design: Three Methods on smarties.png\n"
    "Original  |  Method 1 (Otsu)  |  Method 2 (HSV)  |  Method 3 (Canny)",
    color="white", fontsize=12, fontweight="bold", y=1.04,
)
plt.tight_layout()
plt.savefig("outputs/task10_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 10 -- SEGMENTATION SYSTEM DESIGN: ANALYSIS REPORT")
print("=" * 70)
print()
print("IMAGE: smarties.png")
print("  A tray of multi-coloured candy discs (smarties) on a plain background.")
print("  Challenges: touching objects, multiple colours, similar brightness levels.")
print()

print("-" * 70)
print("METHOD 1 -- OTSU'S GLOBAL THRESHOLDING")
print("-" * 70)
print(f"  Assumption: The image has a bimodal intensity histogram -- one peak")
print(f"  for the background (lower intensity) and one for the candies (higher).")
print(f"  Otsu finds threshold T={int(otsu_val)} to maximise inter-class variance.")
print()
print(f"  Correctly identifies:  The group of candy discs as a whole (foreground).")
print(f"  Incorrectly segments:  Treats ALL candies as a single merged blob;")
print(f"  different-coloured candies with similar intensity are not separated.")
print(f"  Dark-coloured candies (navy, dark brown) may be misclassified as background.")
print()
print(f"  Most impactful parameter: The threshold value T (chosen automatically by Otsu).")
print(f"  Preprocessing (e.g., blurring) also has significant effect.")
print()

print("-" * 70)
print("METHOD 2 -- HSV COLOUR-BASED THRESHOLDING")
print("-" * 70)
print(f"  Assumption: The target object (red smarties) has a distinct hue range")
print(f"  that can be isolated with HSV lower/upper bounds.")
print()
print(f"  Correctly identifies:  Red smarties are precisely isolated from the")
print(f"  rest of the scene, including other colours and the background.")
print(f"  Correctly rejects:     Blue, yellow, green, and purple candies.")
print()
print(f"  Incorrectly segments:  Any red-tinted background area or shadows that")
print(f"  have high red hue values. Also fails completely for non-red candies.")
print(f"  Each colour requires separate tuning of its own HSV range.")
print()
print(f"  Most impactful parameter: The Hue range [H_min, H_max].")
print(f"  A 5-unit error in H_min/H_max can cause major false positives or")
print(f"  false negatives, especially for colours near hue boundaries.")
print()

print("-" * 70)
print("METHOD 3 -- CANNY EDGE DETECTION")
print("-" * 70)
print(f"  Assumption: Boundaries between candies and background are marked by")
print(f"  sharp changes in intensity (high gradient magnitude).")
print()
print(f"  Correctly identifies:  The outer circular boundaries of individual")
print(f"  candy discs. Also captures some internal colour boundaries between")
print(f"  differently coloured touching candies.")
print()
print(f"  Incorrectly segments:  Does not produce a filled binary mask --")
print(f"  only boundaries are detected, not the filled regions. Interior")
print(f"  texture edges (print on candy surface) create unwanted edges.")
print()
print(f"  Most impactful parameter: The HIGH threshold (hysteresis upper bound).")
print(f"  Increasing it reduces edge count significantly, dropping weak boundaries.")
print()

print("-" * 70)
print("TECHNICAL CONCLUSION (based on observed results)")
print("-" * 70)
print()
print("  For smarties.png, METHOD 2 (HSV Colour Thresholding) produces the")
print("  most meaningful segmentation result for the task of isolating a")
print("  specific object.")
print()
print("  Reason: The defining property of smarties is their COLOUR, not their")
print("  grayscale intensity. Multiple candies of different colours can have")
print("  very similar brightness, making intensity-based methods (Otsu) unable")
print("  to separate them individually. Canny detects boundaries but cannot")
print("  fill or label regions.")
print()
print("  HSV thresholding directly exploits the most discriminative feature")
print("  of this image -- the hue of each candy -- to produce clean, filled,")
print("  object-level masks without requiring learning or complex pipelines.")
print()
print("  However, it is colour-specific: a general-purpose system would need")
print("  to run one HSV pass per target colour. For scenarios where object")
print("  separation (not colour isolation) is the goal, watershed (Task 7)")
print("  would be the preferred approach for this image.")
print("=" * 70)
print("\nOutput saved -> outputs/task10_output.png")

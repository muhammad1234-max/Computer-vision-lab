"""
task1.py
Task 1 - Automated Inspection of a Document Under Uneven Lighting

Approach:
  - Load sudoku.png in grayscale (classic unevenly-lit document).
  - Apply global thresholding at three threshold values (80, 127, 180).
  - Apply adaptive Gaussian thresholding.
  - Produce a side-by-side comparison and explain why global thresholding fails.

Parameters selected:
  T1 = 80   -- deliberately low  (captures shadow-side but adds bright-side noise)
  T2 = 127  -- standard midpoint (often used as "safe default")
  T3 = 180  -- deliberately high (cleans bright side, loses dark-side text)
  Adaptive: GAUSSIAN_C, blockSize=11, C=2  -- best overall document result
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

# -- 1. Load image ------------------------------------------------------------
img = cv2.imread("images/sudoku.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("images/sudoku.png not found. Run download_images.py first.")

print(f"Image shape: {img.shape}  |  dtype: {img.dtype}")
print(f"Intensity range: {img.min()} - {img.max()}")

# -- 2. Global thresholding (three values) ------------------------------------
_, g80  = cv2.threshold(img,  80, 255, cv2.THRESH_BINARY)
_, g127 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, g180 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# -- 3. Adaptive thresholding (Gaussian, blockSize=11, C=2) -------------------
adaptive = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2,
)

# -- 4. Comparison plot: Original -> Global T1 -> Global T2 -> Global T3 -> Adaptive
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.patch.set_facecolor("#1a1a2e")

panels = [
    (img,      "Original\n(Grayscale)"),
    (g80,      "Global T = 80\n(Too Low)"),
    (g127,     "Global T = 127\n(Standard)"),
    (g180,     "Global T = 180\n(Too High)"),
    (adaptive, "Adaptive\n(Gaussian, BS=11, C=2)"),
]

for ax, (image, title) in zip(axes, panels):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a90d9")
        spine.set_linewidth(1.5)

fig.suptitle(
    "Task 1 - Document Thresholding Under Uneven Lighting\n"
    "Original  ->  Global T1  ->  Global T2  ->  Global T3  ->  Adaptive",
    color="white", fontsize=13, fontweight="bold", y=1.03,
)
plt.tight_layout()
plt.savefig("outputs/task1_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- 5. Console report --------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 1 -- SELECTED PARAMETERS AND JUSTIFICATION")
print("=" * 65)
print("  T1 = 80  (low threshold)")
print("    Justification: Captures text in darker areas but bright background")
print("    pixels whose intensity < 80 merge with text -> noisy mask.")
print()
print("  T2 = 127  (mid / standard threshold)")
print("    Justification: The 'safe' choice; works on evenly-lit images but")
print("    on this image the dark side has text pixels near or above 127,")
print("    so they are classified as background (characters disappear).")
print()
print("  T3 = 180  (high threshold)")
print("    Justification: Cleans the bright side effectively but almost all")
print("    dark-region text pixels are below 180 and lost entirely.")
print()
print("  Adaptive (Gaussian, blockSize=11, C=2)  <- BEST RESULT")
print("    blockSize=11 : neighbourhood wide enough to capture local contrast")
print("                   without being so large it loses the illumination gradient.")
print("    C=2          : small positive constant that fine-tunes the local")
print("                   threshold downward, preventing over-thresholding of")
print("                   faint strokes.")
print()
print("WHY A SINGLE GLOBAL THRESHOLD FAILS")
print("-" * 65)
print("The sudoku image has a pronounced illumination gradient: the left half")
print("of the page is noticeably brighter than the right half. Any single")
print("threshold T must be set relative to one brightness level:")
print()
print("  - In bright regions, background pixels have high intensity. A low T")
print("    keeps them as background correctly, but a high T merges them with")
print("    foreground (noise).")
print("  - In dark regions, text pixels have lower intensity. A high T")
print("    correctly labels them as background if their intensity < T, but")
print("    that means the text disappears. A low T keeps them, but then the")
print("    bright-region background bleeds through.")
print()
print("There is no single T that simultaneously satisfies both constraints.")
print("Adaptive thresholding solves this by computing an independent local")
print("threshold for every pixel based on the statistics of its neighbourhood,")
print("naturally compensating for the global intensity gradient.")
print("=" * 65)
print("\nOutput saved -> outputs/task1_output.png")

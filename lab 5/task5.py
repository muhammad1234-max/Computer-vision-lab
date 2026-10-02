"""
task5.py
Task 5 - Detecting the Boundary of a Manufactured Part (Canny Edge Detection)

Image: smarties.png (clear circular boundaries, suitable for edge analysis).

Three threshold pairs tested (low threshold fixed, high threshold varies):
  Pair 1 : low=50, high=100  -- low high-T  -> many weak edges promoted
  Pair 2 : low=50, high=150  -- medium       -> balanced result
  Pair 3 : low=50, high=200  -- high high-T  -> only strongest edges survive

Two-threshold hysteresis explained:
  - Pixels > high_T  : definite edge (strong edge)
  - Pixels < low_T   : discarded (not an edge)
  - low_T <= pixel <= high_T : kept only if connected to a strong-edge pixel (weak edge)
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

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Optional: mild Gaussian blur to reduce noise before edge detection
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

print(f"Image shape: {img_bgr.shape}")

# -- Three Canny threshold pairs -----------------------------------------------
PAIRS = [
    (50, 100, "Pair 1\nlow=50, high=100"),
    (50, 150, "Pair 2\nlow=50, high=150"),
    (50, 200, "Pair 3\nlow=50, high=200"),
]

edges = []
for low, high, label in PAIRS:
    e = cv2.Canny(img_blur, low, high)
    edges.append((e, label, low, high))
    n_edge_pixels = np.count_nonzero(e)
    print(f"  {label.replace(chr(10),' ')} -> {n_edge_pixels:,} edge pixels")

# -- Output figure: Original -> Edge 1 -> Edge 2 -> Edge 3 ----------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.patch.set_facecolor("#0d1117")

axes[0].imshow(img_rgb)
axes[0].set_title("Original\n(Colour Input)", color="white", fontsize=11, fontweight="bold")
axes[0].axis("off")

colours = ["#4fc3f7", "#81c995", "#ffd700"]
for ax, (edge_img, label, low, high), col in zip(axes[1:], edges, colours):
    ax.imshow(edge_img, cmap="gray")
    n = np.count_nonzero(edge_img)
    ax.set_title(f"{label}\n{n:,} edge pixels", color=col,
                 fontsize=10, fontweight="bold")
    ax.axis("off")

fig.suptitle(
    "Task 5 - Canny Edge Detection  (low threshold fixed = 50)\n"
    "Original  ->  Edge Result 1  ->  Edge Result 2  ->  Edge Result 3",
    color="white", fontsize=13, fontweight="bold", y=1.03,
)
plt.tight_layout()
plt.savefig("outputs/task5_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 5 -- CANNY EDGE DETECTION REPORT")
print("=" * 65)
print()
print("Two-threshold hysteresis (from the manual):")
print("  Gradient magnitude > high_T        -> STRONG EDGE  (always kept)")
print("  low_T <= gradient <= high_T           -> WEAK EDGE   (kept only if")
print("                                         connected to a strong edge)")
print("  Gradient magnitude < low_T          -> NOT AN EDGE (discarded)")
print()
print("PAIR 1  (low=50, high=100)  -- Low high-T")
print("  Strong edges : pixels with gradient > 100 (many qualify)")
print("  Weak edges   : pixels 50-100 promoted if connected to strong edges")
print("  Effect       : Many edges detected, including weaker texture")
print("                 edges and some noise-induced false edges inside")
print("                 homogeneous regions. Most boundary edges captured.")
print("  Verdict      : Rich but noisy -- unwanted edges inside candies.")
print()
print("PAIR 2  (low=50, high=150)  -- Medium (balanced)")
print("  Strong edges : pixels with gradient > 150 -- fewer qualify.")
print("  Fewer weak edges promoted (smaller hysteresis chain).")
print("  Effect       : Clean boundary detection; interior texture edges")
print("                 largely suppressed while candy outlines remain.")
print("  Verdict      : Best overall balance for manufactured-part inspection.")
print()
print("PAIR 3  (low=50, high=200)  -- High high-T")
print("  Strong edges : only very steep gradients (> 200) survive.")
print("  Many genuine boundary edges fall below 200 and cannot anchor")
print("  the hysteresis chain -> those segments simply vanish.")
print("  Effect       : Very selective -- only sharpest edges survive.")
print("                 Some candy boundaries are MISSING (broken contours).")
print("  Verdict      : Under-detection; useful only for finding dominant")
print("                 structural boundaries, not complete outlines.")
print()
print("EDGE TAXONOMY FOR THIS IMAGE:")
print("  Strong edges  : Sharpest ring boundaries between candies + background.")
print("  Weak edges    : Softer colour transitions inside candy clusters.")
print("  Missing edges : Boundaries lost at high high-T (Pair 3).")
print("  Unwanted edges: Texture/shadow inside candies at low high-T (Pair 1).")
print()
print("HOW CHANGING THRESHOLDS AFFECTS THE RESULT:")
print("  Increasing high_T -> reduces the set of strong seed edges.")
print("  Without strong seeds, adjacent weak edges are not promoted.")
print("  Result: contours become fragmented or entirely disappear.")
print("  The LOW threshold should remain ~= high_T/3 (Canny's recommendation).")
print("  Fixing low=50 and raising high forces the ratio to deviate,")
print("  demonstrating that high_T is the dominant control parameter.")
print("=" * 65)
print("\nOutput saved -> outputs/task5_output.png")

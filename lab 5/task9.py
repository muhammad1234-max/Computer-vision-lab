"""
task9.py
Task 9 - Segmenting an Image Using K-Means Clustering

Image: dog.jpeg (complex colour photograph -- wildlife / natural scene).

Procedure (from the manual -- K-Means workflow):
  1. Load colour image.
  2. Reshape pixel array to a flat list of (B, G, R) feature vectors.
  3. Convert to float32.
  4. Run cv2.kmeans to cluster pixels.
  5. Map cluster centres back to pixel positions -> reconstructed image.

K values tested: 2, 4, 6

For each K record:
  - Number of clusters
  - Visual difference from original
  - Amount of colour simplification
  - Major regions represented by the clusters
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

# -- 1. Load colour image ------------------------------------------------------
img_bgr = cv2.imread("images/dog.jpeg")
if img_bgr is None:
    img_bgr = cv2.imread("images/dog.jpg")
if img_bgr is None:
    # Fallback: use smarties.png as complex colour image
    print("WARNING: dog.jpeg not found -- using smarties.png as fallback.")
    img_bgr = cv2.imread("images/smarties.png")
if img_bgr is None:
    raise FileNotFoundError(
        "Neither dog.jpeg nor smarties.png found. Run download_images.py first."
    )

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"Image shape: {img_bgr.shape}")

# -- K-Means segmentation function ---------------------------------------------
def kmeans_segment(image_bgr, k, attempts=10):
    """
    Segment image_bgr using K-Means clustering on pixel colour vectors.
    Returns the reconstructed (quantised) image in BGR.
    """
    # Step 2: Reshape to flat pixel list   shape: (H*W, 3)
    pixel_data = image_bgr.reshape((-1, 3))

    # Step 3: Convert to float32
    pixel_data = np.float32(pixel_data)

    # Step 4: K-Means clustering
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,    # max iterations
        0.2,    # epsilon
    )
    _, labels, centres = cv2.kmeans(
        pixel_data,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,    # K-Means++ initialisation
    )

    # Step 5: Reconstruct image from cluster centres
    centres   = np.uint8(centres)
    segmented = centres[labels.flatten()]
    segmented = segmented.reshape(image_bgr.shape)

    return segmented, labels.reshape(image_bgr.shape[:2]), centres

# -- Run K = 2, 4, 6 -----------------------------------------------------------
K_VALUES = [2, 4, 6]
results  = {}

for k in K_VALUES:
    seg_bgr, label_map, centres_bgr = kmeans_segment(img_bgr, k)
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    results[k] = (seg_rgb, label_map, centres_bgr)

    # Colour simplification: unique colours in result
    unique_colours = np.unique(seg_bgr.reshape(-1, 3), axis=0)
    print(f"  K={k}: {len(unique_colours)} unique colours in output "
          f"(reduced from millions)")

# -- Output figure: Original -> K=2 -> K=4 -> K=6 -------------------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.patch.set_facecolor("#0d1117")

axes[0].imshow(img_rgb)
axes[0].set_title("Original\n(Full Colour Range)", color="white",
                  fontsize=11, fontweight="bold")
axes[0].axis("off")

k_colours = ["#ffd700", "#81c995", "#4fc3f7"]
for ax, k, col in zip(axes[1:], K_VALUES, k_colours):
    seg_rgb, label_map, centres_bgr = results[k]

    # Compute pixel-level colour error vs original
    diff = np.mean(np.abs(img_rgb.astype(np.float32) -
                          seg_rgb.astype(np.float32)))

    ax.imshow(seg_rgb)
    ax.set_title(
        f"K = {k}\nMean colour error: {diff:.1f}",
        color=col, fontsize=10, fontweight="bold",
    )
    ax.axis("off")

fig.suptitle(
    "Task 9 - K-Means Image Segmentation\n"
    "Original  ->  K=2  ->  K=4  ->  K=6",
    color="white", fontsize=13, fontweight="bold", y=1.03,
)
plt.tight_layout()
plt.savefig("outputs/task9_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close("all")

# -- Cluster colour palette figure --------------------------------------------
fig2, axes2 = plt.subplots(len(K_VALUES), 1, figsize=(12, 5))
fig2.patch.set_facecolor("#0d1117")
if len(K_VALUES) == 1:
    axes2 = [axes2]

for ax, k in zip(axes2, K_VALUES):
    _, _, centres_bgr = results[k]
    centres_rgb = centres_bgr[:, ::-1]   # BGR -> RGB
    palette = centres_rgb.reshape(1, k, 3).astype(np.uint8)
    ax.imshow(np.repeat(palette, 50, axis=0))
    ax.set_title(f"K={k}  Cluster Colour Palette",
                 color="white", fontsize=10, fontweight="bold")
    for j, c in enumerate(centres_rgb):
        ax.text(j, 25, f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}",
                ha="center", va="center", fontsize=7,
                color="white" if c.mean() < 128 else "black", fontweight="bold")
    ax.axis("off")

fig2.suptitle("Task 9 - Cluster Colour Palettes", color="white",
              fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/task9_palettes.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.close("all")

# -- Report ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("TASK 9 -- K-MEANS SEGMENTATION REPORT")
print("=" * 65)
print()
print("K = 2  (2 clusters)")
print("  The entire image is reduced to 2 dominant colours.")
print("  Usually separates bright vs. dark regions (e.g., sky vs. ground,")
print("  or fur vs. background). Extremely coarse -- fine details lost.")
print("  Use case: rough foreground/background separation.")
print()
print("K = 4  (4 clusters)")
print("  Introduces enough clusters to distinguish major colour regions:")
print("  e.g., different fur tones, grass, sky, shadow areas.")
print("  A good middle ground between colour fidelity and simplicity.")
print("  Fine texture and gradients are still lost.")
print()
print("K = 6  (6 clusters)")
print("  More nuanced segmentation; subtle colour differences start to")
print("  appear as separate regions. Closer to the original appearance.")
print("  Some fine edges and small regions begin to be correctly isolated.")
print("  However, visually distinct objects may still share a cluster.")
print()
print("PROCEDURE USED (from manual):")
print("  reshape pixels -> convert float32 -> cv2.kmeans (K-Means++)")
print("  -> map labels to centres -> reconstruct image.")
print()
print("COLOUR SIMPLIFICATION:")
print("  K=2 : Reduces image to 2 colours   -> maximum simplification.")
print("  K=4 : 4 dominant palette colours   -> moderate simplification.")
print("  K=6 : 6 palette colours            -> least simplification, closest")
print("        to original appearance.")
print()
print("The K-Means++ initialisation (KMEANS_PP_CENTERS) was used to avoid")
print("poor local minima that random initialisation can produce.")
print("=" * 65)
print("\nOutputs saved ->")
print("  outputs/task9_output.png")
print("  outputs/task9_palettes.png")

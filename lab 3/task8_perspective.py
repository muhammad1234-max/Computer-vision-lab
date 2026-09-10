"""
Task 8: The Sidewalk Illusion (Projective - Perspective)

Scenario:
    You took a photo of 3D chalk art on a sidewalk, but because you were
    standing in front of it, the far edge looks smaller than the close edge.

Task:
    - Pick 4 corners of the art that should mathematically form a perfect square.
    - Build a perspective transformation matrix to warp the image into a
      top-down "birds-eye" view.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)


def make_chalk_art(size=400):
    """
    Creates a synthetic chalk-art image on a textured sidewalk background.
    The artwork is a colourful spiral / mandala pattern drawn on pavement.
    """
    img = np.full((size, size, 3), (160, 155, 148), dtype=np.uint8)  # concrete grey

    # Sidewalk texture (random lighter patches)
    rng = np.random.default_rng(42)
    for _ in range(180):
        px = rng.integers(0, size)
        py = rng.integers(0, size)
        r  = rng.integers(3, 20)
        v  = int(rng.integers(-15, 15))
        cv2.circle(img, (px, py), r, (160 + v, 155 + v, 148 + v), -1)

    # Draw the chalk artwork in the center (square region)
    cx, cy, art_r = size // 2, size // 2, size // 3

    # Outer ring
    cv2.circle(img, (cx, cy), art_r, (80, 60, 200), 4)
    # Colourful sectors
    colors = [(220, 80, 80), (80, 200, 80), (80, 80, 220),
              (220, 200, 60), (200, 80, 200), (60, 200, 200)]
    for i, color in enumerate(colors):
        start_angle = i * 60
        end_angle   = start_angle + 55
        cv2.ellipse(img, (cx, cy), (art_r - 10, art_r - 10),
                    0, start_angle, end_angle, color, 12)

    # Inner pattern
    for r in range(20, art_r - 15, 30):
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)
    cv2.circle(img, (cx, cy), 18, (240, 200, 50), -1)
    cv2.circle(img, (cx, cy), 8,  (255, 100, 50), -1)

    return img


# ── Create the clean (top-down) chalk art ────────────────────────────────────
art_clean = make_chalk_art(400)
H, W = art_clean.shape[:2]

print("=" * 50)
print("Task 8: The Sidewalk Illusion - Projective Perspective")
print("=" * 50)


# ── Step 1: Define the 4 source corners (perspective-distorted view) ──────────
#
# Imagine standing in front of the square chalk art at a low angle.
# The far edge (top of image) appears narrower than the near edge (bottom).
# These 4 points represent the trapezoid the square looks like in the photo.
#
# We'll simulate this by defining a trapezoidal quad as our "photo corners"
# and a perfect rectangle as the "true" corners.

margin = 20
# Destination: perfect rectangle (what it SHOULD look like from above)
dst_pts = np.float32([
    [margin,       margin      ],   # top-left
    [W - margin,   margin      ],   # top-right
    [W - margin,   H - margin  ],   # bottom-right
    [margin,       H - margin  ],   # bottom-left
])

# Source: trapezoidal (perspective-distorted) quad
# Far edge is narrower (converging toward vanishing point)
src_pts = np.float32([
    [80,       60 ],    # top-left  (inward - far edge narrower)
    [320,      60 ],    # top-right (inward - far edge narrower)
    [380,      340],    # bottom-right (wider near edge)
    [20,       340],    # bottom-left  (wider near edge)
])

print("\nPerspective correction:")
print("  Source points (trapezoid in photo - perspective view):")
for i, pt in enumerate(src_pts):
    print(f"    {i}: ({pt[0]:.0f}, {pt[1]:.0f})")
print("  Destination points (perfect rectangle - birds-eye view):")
for i, pt in enumerate(dst_pts):
    print(f"    {i}: ({pt[0]:.0f}, {pt[1]:.0f})")


# Apply the inverse to create the "photo" version (rectangle -> trapezoid)
# This simulates taking a photo of the art from an angle
M_photo = cv2.getPerspectiveTransform(dst_pts, src_pts)
art_photo = cv2.warpPerspective(art_clean, M_photo, (W, H),
                                borderValue=(160, 155, 148))

# Draw the 4 corner points on the photo version
art_photo_marked = art_photo.copy()
corner_colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200)]
labels = ["TL", "TR", "BR", "BL"]
for pt, col, lbl in zip(src_pts, corner_colors, labels):
    cv2.circle(art_photo_marked, (int(pt[0]), int(pt[1])), 7, col, -1)
    cv2.putText(art_photo_marked, lbl, (int(pt[0]) + 8, int(pt[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


# ── Step 2: Compute the Perspective (Homography) matrix ──────────────────────
#
# cv2.getPerspectiveTransform() needs exactly 4 point correspondences.
# It solves for the 3x3 Homography matrix H such that:
#
#   [x'] = H [x]   (in homogeneous coords)
#   [y']   H [y]
#   [w']     [1]
#
# Final pixel: x_out = x'/w',  y_out = y'/w'
# The division by w' is what makes it a TRUE perspective transform
# (unlike affine which always has w'=1).

M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("\nComputed 3x3 Perspective (Homography) Matrix H:")
print(np.round(M_perspective, 6))
print("\nNote: the bottom row [h7, h8, 1] makes this a projective transform.")
print("The division by w' = h7*x + h8*y + 1 creates the depth foreshortening effect.")


# ── Step 3: Apply the perspective warp ────────────────────────────────────────
art_corrected = cv2.warpPerspective(art_photo, M_perspective, (W, H),
                                    flags=cv2.INTER_LINEAR)


# ── Visualize ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Task 8: The Sidewalk Illusion - Projective Perspective Transform',
             fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(art_clean, cv2.COLOR_BGR2RGB))
axes[0].set_title("True Chalk Art\n(perfect square, bird's-eye view)")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(art_photo_marked, cv2.COLOR_BGR2RGB))
axes[1].set_title("Photo from Sidewalk\n(trapezoid - 4 corners marked)")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(art_corrected, cv2.COLOR_BGR2RGB))
axes[2].set_title("Corrected (Bird's-Eye)\n(perspective warp applied)")
axes[2].axis('off')

fig.text(0.5, 0.01,
         "Homography H: 3x3 matrix  |  warpPerspective divides by w'=h7*x+h8*y+1",
         ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('output/task8_perspective.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task8_perspective.png")

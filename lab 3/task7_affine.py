"""
Task 7: The Glitched Image (General Affine Transformation)

Scenario:
    A corrupted data transmission resulted in a photograph being squished,
    slanted, rotated, and moved all at once.

Task:
    - You are given the starting and ending coordinates of three distinct
      landmarks in the photo.
    - Solve a system of linear equations using these 3 point pairs to find
      the 6 unknown values of a general Affine matrix.
    - Apply the matrix to fix the image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)


def make_photo(width=300, height=220):
    """
    Creates a synthetic 'photograph' with identifiable landmark features.
    """
    img = np.full((height, width, 3), (200, 195, 185), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 2):
        val = int(135 + y * 0.6)
        img[y, :] = (val, val + 30, min(255, val + 80))

    # Ground
    img[height // 2:, :] = (100, 140, 80)

    # Building (landmark A)
    cv2.rectangle(img, (40, 60), (120, height // 2), (160, 150, 140), -1)
    cv2.rectangle(img, (40, 60), (120, height // 2), (80, 70, 60), 2)
    # Windows
    for wy in range(75, height // 2 - 15, 25):
        for wx in range(50, 115, 20):
            cv2.rectangle(img, (wx, wy), (wx + 12, wy + 16), (180, 210, 240), -1)

    # Tower (landmark B)
    cv2.rectangle(img, (160, 30), (200, height // 2), (140, 135, 130), -1)
    cv2.rectangle(img, (160, 30), (200, height // 2), (60, 55, 50), 2)
    cv2.line(img, (180, 5), (180, 30), (80, 80, 80), 3)  # antenna

    # Tree (landmark C)
    cv2.circle(img, (250, 90), 30, (40, 110, 40), -1)
    cv2.rectangle(img, (244, 90), (258, height // 2), (90, 60, 40), -1)

    # Draw landmark markers with labels
    for pt, label, color in [((80, 60), "A", (0, 0, 220)),
                              ((180, 30), "B", (0, 200, 0)),
                              ((250, 60), "C", (220, 100, 0))]:
        cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, label, (pt[0] + 6, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return img


# Create clean photo and apply a "glitch" affine distortion
photo_clean = make_photo()
H, W = photo_clean.shape[:2]

# Known glitch transform (this is what we will RECOVER from point correspondences)
# General Affine has 6 degrees of freedom: a, b, c, d, tx, ty
# x' = a*x + b*y + tx
# y' = c*x + d*y + ty
a_true, b_true = 0.8, 0.2
c_true, d_true = -0.15, 0.9
tx_true, ty_true = 40.0, 30.0

M_glitch = np.float32([
    [a_true, b_true, tx_true],
    [c_true, d_true, ty_true]
])

photo_glitched = cv2.warpAffine(photo_clean, M_glitch, (W, H),
                                flags=cv2.INTER_LINEAR, borderValue=(180, 180, 180))

print("=" * 50)
print("Task 7: The Glitched Image - General Affine Transformation")
print("=" * 50)
print("\nThe image was corrupted: squished, slanted, rotated, and translated.")
print("We have 3 landmark point correspondences to recover the affine matrix.\n")


# --- Step 1: Define the 3 corresponding landmark points ---
#
# We know where three points ARE in the glitched image (src_pts)
# and where they SHOULD BE (dst_pts = original locations).
#
# The three landmarks: A (building top), B (tower top), C (tree top)

src_pts = np.array([
    [80,  60],    # Landmark A in clean image
    [180, 30],    # Landmark B in clean image
    [250, 60],    # Landmark C in clean image
], dtype=np.float64)

# Apply the known glitch matrix to find where they ended up
def apply_affine(M, pts):
    """Apply a 2x3 affine matrix to a set of points (Nx2)."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # Nx3 homogeneous
    return (M @ pts_h.T).T  # Nx2

dst_pts = apply_affine(M_glitch, src_pts)  # where landmarks ended up after glitch

print("Landmark correspondences (3 point pairs):")
for i, (s, d) in enumerate(zip(src_pts, dst_pts)):
    label = chr(65 + i)
    print(f"  {label}: clean ({s[0]:.0f},{s[1]:.0f})  ->  glitched ({d[0]:.1f},{d[1]:.1f})")


# --- Step 2: Solve the system of linear equations ---
#
# We want a matrix M_fix that maps glitched -> clean.
# For each correspondence (xd_i, yd_i) [glitched] -> (xc_i, yc_i) [clean]:
#   xc_i = a * xd_i + b * yd_i + tx
#   yc_i = c * xd_i + d * yd_i + ty
#
# That gives 2 equations per point. With 3 points we get 6 equations
# for 6 unknowns: [a, b, tx] and [c, d, ty].
#
# Matrix form (for x-channel):
#   [ xd1  yd1  1 ]   [ a  ]   [ xc1 ]
#   [ xd2  yd2  1 ] * [ b  ] = [ xc2 ]
#   [ xd3  yd3  1 ]   [ tx ]   [ xc3 ]
#
# Solve with numpy.linalg.solve
#
# dst_pts = glitched coords (input to the system)
# src_pts = clean coords   (right-hand side / target)

A_mat = np.array([
    [dst_pts[0, 0], dst_pts[0, 1], 1],
    [dst_pts[1, 0], dst_pts[1, 1], 1],
    [dst_pts[2, 0], dst_pts[2, 1], 1],
], dtype=np.float64)

# Right-hand sides: clean (target) x and y coordinates
rhs_x = src_pts[:, 0]   # clean x: [80, 180, 250]
rhs_y = src_pts[:, 1]   # clean y: [60, 30,  60 ]

# Solve for [a, b, tx] and [c, d, ty] separately
params_x = np.linalg.solve(A_mat, rhs_x)  # [a, b, tx]
params_y = np.linalg.solve(A_mat, rhs_y)  # [c, d, ty]

a_r,  b_r,  tx_r  = params_x
c_r,  d_r,  ty_r  = params_y

# The inverse of the glitch matrix (true fix) for comparison
M_glitch_3x3 = np.vstack([M_glitch, [0, 0, 1]])
M_fix_true_3x3 = np.linalg.inv(M_glitch_3x3)
M_fix_true = M_fix_true_3x3[:2, :]

print("\nSolved Affine FIX matrix (glitched -> clean), from 3-point correspondences:")
print(f"  Row 1 (x'): a={a_r:.4f}, b={b_r:.4f}, tx={tx_r:.4f}")
print(f"  Row 2 (y'): c={c_r:.4f}, d={d_r:.4f}, ty={ty_r:.4f}")
print("\nTrue inverse of glitch matrix (for reference):")
print(f"  Row 1: a={M_fix_true[0,0]:.4f}, b={M_fix_true[0,1]:.4f}, tx={M_fix_true[0,2]:.4f}")
print(f"  Row 2: c={M_fix_true[1,0]:.4f}, d={M_fix_true[1,1]:.4f}, ty={M_fix_true[1,2]:.4f}")


# --- Step 3: Build and apply the correction matrix ---
#
# The solved matrix M_inv undoes the glitch (maps glitched -> clean).

M_fix = np.float32([
    [a_r, b_r, tx_r],
    [c_r, d_r, ty_r]
])

photo_fixed = cv2.warpAffine(photo_glitched, M_fix, (W, H),
                              flags=cv2.INTER_LINEAR, borderValue=(180, 180, 180))


# --- Visualize ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Task 7: The Glitched Image - General Affine Transformation',
             fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(photo_clean, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Clean Photo\n(3 landmarks marked A, B, C)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(photo_glitched, cv2.COLOR_BGR2RGB))
axes[1].set_title('Glitched Photo\n(squished, slanted, rotated, moved)')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(photo_fixed, cv2.COLOR_BGR2RGB))
axes[2].set_title('Corrected Photo\n(affine matrix solved from 3 points)')
axes[2].axis('off')

matrix_str = (f"Solved: [[{a_r:.3f}, {b_r:.3f}, {tx_r:.1f}], "
              f"[{c_r:.3f}, {d_r:.3f}, {ty_r:.1f}]]  |  6 unknowns, 6 equations")
fig.text(0.5, 0.01, matrix_str, ha='center', fontsize=8, color='steelblue')

plt.tight_layout()
plt.savefig('output/task7_affine.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task7_affine.png")

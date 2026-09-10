"""
Task 9: The Stadium Panorama (Projective - Homography)

Scenario:
    You have two photos from different cameras covering a football field.
    They share a small overlapping area in the middle.

Task:
    - Select 4 matching points found in both images.
    - Calculate the Homography matrix to mathematically project the second
      camera's view into the first camera's space, stitching them together
      into a seamless panorama.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)


def make_stadium_half(width=500, height=350, left_half=True):
    """
    Creates one half of a football field viewed from above.
    Field markings (lines, circles, penalty box) included.
    """
    # Green field
    img = np.full((height, width, 3), (40, 130, 40), dtype=np.uint8)

    # Alternating darker stripes (mowing pattern)
    stripe_w = 50
    for i in range(0, width, stripe_w * 2):
        cv2.rectangle(img, (i, 0), (i + stripe_w, height), (35, 118, 35), -1)

    # Field boundary lines
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (255, 255, 255), 2)

    # Halfway line (right edge for left half, left edge for right half)
    if left_half:
        cv2.line(img, (width - 10, 10), (width - 10, height - 10), (255, 255, 255), 2)
        # Centre circle (half visible)
        cv2.ellipse(img, (width - 10, height // 2), (60, 60), 0, 90, 270,
                    (255, 255, 255), 2)
        # Penalty box
        cv2.rectangle(img, (10, height // 2 - 70), (130, height // 2 + 70),
                      (255, 255, 255), 2)
        # Goal area
        cv2.rectangle(img, (10, height // 2 - 35), (55, height // 2 + 35),
                      (255, 255, 255), 2)
        # Penalty spot
        cv2.circle(img, (100, height // 2), 5, (255, 255, 255), -1)
        # Goal
        cv2.rectangle(img, (0, height // 2 - 45), (10, height // 2 + 45),
                      (255, 255, 255), 3)
        # Camera label
        cv2.putText(img, "CAM A", (180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
    else:
        cv2.line(img, (10, 10), (10, height - 10), (255, 255, 255), 2)
        # Centre circle (half visible)
        cv2.ellipse(img, (10, height // 2), (60, 60), 0, -90, 90,
                    (255, 255, 255), 2)
        # Penalty box
        cv2.rectangle(img, (width - 130, height // 2 - 70),
                      (width - 10, height // 2 + 70), (255, 255, 255), 2)
        # Goal area
        cv2.rectangle(img, (width - 55, height // 2 - 35),
                      (width - 10, height // 2 + 35), (255, 255, 255), 2)
        # Penalty spot
        cv2.circle(img, (width - 100, height // 2), 5, (255, 255, 255), -1)
        # Goal
        cv2.rectangle(img, (width - 10, height // 2 - 45),
                      (width, height // 2 + 45), (255, 255, 255), 3)
        # Camera label
        cv2.putText(img, "CAM B", (270, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)

    return img


# ── Create both camera views ──────────────────────────────────────────────────
IMG_W, IMG_H = 500, 350
cam_A = make_stadium_half(IMG_W, IMG_H, left_half=True)
cam_B = make_stadium_half(IMG_W, IMG_H, left_half=False)

# Simulate Camera B having a slight perspective shift (different mounting angle)
# Apply a small known homography to cam_B to simulate real-world misalignment
M_shift = np.float64([
    [1.02,  0.03,  -8.0],
    [0.01,  0.98,   5.0],
    [0.00005, 0.0001, 1.0]
])
cam_B_shifted = cv2.warpPerspective(cam_B, M_shift, (IMG_W, IMG_H),
                                    borderValue=(40, 130, 40))

print("=" * 50)
print("Task 9: The Stadium Panorama - Projective Homography")
print("=" * 50)


# ── Step 1: Select 4 matching point pairs in the overlap region ───────────────
#
# These are points visible in BOTH camera images (the overlap zone near centre).
# Camera A's right edge overlaps with Camera B's left edge.
#
# Points chosen on field markings visible in both views:
#   - Corner of the centre circle arc at different y positions
#   - Points on the halfway line

# Points in Camera A (source)
pts_A = np.float32([
    [IMG_W - 10, IMG_H // 2 - 60],    # top of centre circle
    [IMG_W - 10, IMG_H // 2 + 60],    # bottom of centre circle
    [IMG_W - 60, IMG_H // 2      ],    # left of centre circle
    [IMG_W - 10, IMG_H // 2      ],    # centre of halfway line
])

# Corresponding points in Camera B (destination, accounting for shift)
def apply_homography_pts(H, pts):
    """Apply homography H to a set of points (Nx2 float32)."""
    pts_h = np.column_stack([pts, np.ones(len(pts))])   # Nx3
    result = (H @ pts_h.T).T                             # Nx3
    result /= result[:, 2:3]                             # divide by w
    return result[:, :2].astype(np.float32)

pts_B = apply_homography_pts(M_shift,
                             np.float32([
                                 [10, IMG_H // 2 - 60],
                                 [10, IMG_H // 2 + 60],
                                 [60, IMG_H // 2     ],
                                 [10, IMG_H // 2     ],
                             ]))

print("\n4 Matching Point Pairs (overlap region):")
for i, (a, b) in enumerate(zip(pts_A, pts_B)):
    print(f"  Point {i+1}: CamA({a[0]:.0f},{a[1]:.0f})  <->  CamB({b[0]:.1f},{b[1]:.1f})")


# ── Step 2: Calculate the Homography matrix ───────────────────────────────────
#
# cv2.findHomography() uses DLT (Direct Linear Transform) to solve for the
# 3x3 Homography matrix H from N >= 4 point correspondences.
#
# H maps points from Camera B's coordinate space -> Camera A's coordinate space.
# Once we have H, we can warpPerspective cam_B into cam_A's frame and blend them.

H, mask = cv2.findHomography(pts_B, pts_A, method=0)   # method=0: least squares

print("\nCalculated 3x3 Homography Matrix H (B -> A):")
print(np.round(H, 6))
print(f"\nAll 4 points used (mask sum = {mask.sum() if mask is not None else 4})")


# ── Step 3: Warp Camera B into Camera A's space ───────────────────────────────
#
# Create a wide panorama canvas (double width) and place cam_A on the left.
# Warp cam_B using H into this same wide canvas.

pan_w = IMG_W * 2 - 80   # overlap the two halves by ~80 px
pan_h = IMG_H

panorama = np.zeros((pan_h, pan_w, 3), dtype=np.uint8)

# Place cam_A on the left
panorama[:, :IMG_W] = cam_A

# Adjust H to account for the panorama offset (cam_A is placed at x=0)
# cam_B needs to be warped and placed starting around x = IMG_W - 80
H_pan = H.copy()

# Warp cam_B into panorama space
cam_B_warped = cv2.warpPerspective(cam_B_shifted, H_pan, (pan_w, pan_h),
                                   flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

# Blend in the warped cam_B (simple alpha blend in overlap zone)
# Find where cam_B_warped has valid (non-zero) pixels
mask_B = (cam_B_warped.sum(axis=2) > 0)
# For non-overlapping region of cam_B, just copy
mask_A_only = ~mask_B  # pixels only in cam_A
panorama[mask_B] = cam_B_warped[mask_B]  # cam_B covers non-A area
# Restore cam_A region (left half)
panorama[:, :IMG_W - 80] = cam_A[:, :IMG_W - 80]
# Blend overlap zone
for x in range(IMG_W - 80, IMG_W):
    alpha = (x - (IMG_W - 80)) / 80.0  # 0 at start of overlap, 1 at end
    col_a = cam_A[:, x].astype(np.float32)
    col_b = cam_B_warped[:, x].astype(np.float32)
    valid = cam_B_warped[:, x].sum(axis=1) > 0
    blended = ((1 - alpha) * col_a + alpha * col_b).astype(np.uint8)
    panorama[:, x] = np.where(valid[:, None], blended, cam_A[:, x])


# ── Visualize ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Task 9: The Stadium Panorama - Projective Homography',
             fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(cv2.cvtColor(cam_A, cv2.COLOR_BGR2RGB))
ax1.set_title('Camera A (Left half)')
ax1.axis('off')

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(cv2.cvtColor(cam_B_shifted, cv2.COLOR_BGR2RGB))
ax2.set_title('Camera B (Right half, slightly shifted)')
ax2.axis('off')

ax3 = fig.add_subplot(2, 1, 2)
ax3.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
ax3.set_title(f'Stitched Panorama ({pan_w}x{pan_h}) - Homography H applied to project B into A')
ax3.axis('off')

# Mark the 4 matching points on cam_A
ax1.scatter(pts_A[:, 0], pts_A[:, 1], c='red', s=60, zorder=5)
ax1.set_title('Camera A\n(4 matching points marked red)')

ax2.scatter(pts_B[:, 0], pts_B[:, 1], c='yellow', s=60, zorder=5)
ax2.set_title('Camera B\n(4 matching points marked yellow)')

fig.text(0.5, 0.01,
         "H (3x3 Homography): maps CamB coords -> CamA coords  |  Computed via DLT from 4 point pairs",
         ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('output/task9_homography.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task9_homography.png")

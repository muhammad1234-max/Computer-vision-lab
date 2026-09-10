"""
Task 10: The Master Forger (Full Hierarchy Challenge)

Scenario:
    You need to digitally insert a flat image of a painting into an empty,
    heavily angled picture frame hanging on a museum wall.

Task:
    Traverse the entire transformation hierarchy:
      1. Linear Scale  - shrink the painting to fit inside the frame
      2. Rigid Transform - rotate + translate to move it near the frame
      3. Projective Transform - warp the painting's 4 corners precisely
         into the 3D perspective of the angled frame
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.makedirs('output', exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper images
# ─────────────────────────────────────────────────────────────────────────────

def make_painting(width=300, height=220):
    """Creates a colourful synthetic painting (landscape scene)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (blue -> light blue)
    for y in range(height * 2 // 3):
        t = y / (height * 2 / 3)
        r = int(40  + t * 80)
        g = int(100 + t * 100)
        b = int(200 + t * 40)
        img[y, :] = (b, g, r)   # BGR

    # Ground (green-brown)
    img[height * 2 // 3:, :] = (50, 110, 60)

    # Sun
    cv2.circle(img, (width - 60, 50), 28, (30, 210, 255), -1)
    cv2.circle(img, (width - 60, 50), 32, (50, 180, 230), 3)

    # Mountains
    mountain_pts = np.array([
        [0, height * 2 // 3],
        [60, height // 3],
        [120, height * 2 // 3],
        [180, height // 4],
        [240, height * 2 // 3],
        [width, height * 2 // 3]
    ], dtype=np.int32)
    cv2.fillPoly(img, [mountain_pts], (90, 100, 80))
    cv2.polylines(img, [mountain_pts], False, (200, 210, 200), 2)

    # River
    river_pts = np.array([[110, height * 2 // 3], [130, height - 10],
                           [170, height - 10], [160, height * 2 // 3]], np.int32)
    cv2.fillPoly(img, [river_pts], (150, 160, 80))

    # Tree
    cv2.rectangle(img, (35, height * 2 // 3 - 40), (45, height * 2 // 3), (40, 60, 90), -1)
    cv2.circle(img, (40, height * 2 // 3 - 50), 22, (30, 130, 50), -1)

    # Gold frame border on the painting itself
    cv2.rectangle(img, (3, 3), (width - 3, height - 3), (30, 160, 210), 3)

    return img


def make_museum_wall(width=800, height=600):
    """Creates a museum wall with an angled empty picture frame."""
    # Warm off-white wall
    img = np.full((height, width, 3), (210, 205, 195), dtype=np.uint8)

    # Wall texture
    rng = np.random.default_rng(7)
    for _ in range(300):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        v = int(rng.integers(-8, 8))
        cv2.circle(img, (x, y), rng.integers(2, 12), (210 + v, 205 + v, 195 + v), -1)

    # Floor line
    cv2.line(img, (0, height - 80), (width, height - 80), (170, 155, 140), 3)
    cv2.rectangle(img, (0, height - 80), (width, height), (160, 145, 130), -1)

    # The 4 corners of the empty angled frame on the wall
    # (drawn as a quadrilateral to simulate perspective tilt)
    frame_corners = np.array([
        [270, 80],    # top-left
        [590, 110],   # top-right
        [610, 390],   # bottom-right
        [250, 360],   # bottom-left
    ], dtype=np.int32)

    # Frame backing (dark)
    cv2.fillPoly(img, [frame_corners], (60, 55, 50))

    # Frame border (gold)
    cv2.polylines(img, [frame_corners], True, (30, 160, 210), 18)
    cv2.polylines(img, [frame_corners], True, (20, 110, 160), 2)

    return img, frame_corners.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

painting   = make_painting(300, 220)
museum_wall, frame_corners = make_museum_wall(800, 600)

pH, pW = painting.shape[:2]
wH, wW = museum_wall.shape[:2]

print("=" * 60)
print("Task 10: The Master Forger - Full Transformation Hierarchy")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Linear Scale (2×2 matrix) - shrink the painting
# ─────────────────────────────────────────────────────────────────────────────
#
# Compute how large the frame is and scale the painting to match.
# Frame width ≈ distance between TL and TR corners.

frame_tl, frame_tr, frame_br, frame_bl = frame_corners

frame_w_px = np.linalg.norm(frame_tr - frame_tl)
frame_h_px = np.linalg.norm(frame_bl - frame_tl)

sx = frame_w_px / pW   # uniform scale factors
sy = frame_h_px / pH

# For a proper scale we want a single uniform scale that fits INSIDE the frame
s_uniform = min(sx, sy)

S_2x2 = np.array([
    [s_uniform, 0.0       ],
    [0.0,       s_uniform ]
], dtype=np.float64)

print(f"\nStep 1 - Linear Scale (2x2 matrix):")
print(f"  Frame approx size: {frame_w_px:.0f} x {frame_h_px:.0f} px")
print(f"  Painting size:     {pW} x {pH} px")
print(f"  Scale factor s =   {s_uniform:.4f}")
print(f"  S = {S_2x2.tolist()}")

scaled_w = int(pW * s_uniform)
scaled_h = int(pH * s_uniform)

M_scale = np.float32([[s_uniform, 0, 0], [0, s_uniform, 0]])
painting_scaled = cv2.warpAffine(painting, M_scale, (scaled_w, scaled_h))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Rigid Transformation (3×3) - rotate + translate near frame
# ─────────────────────────────────────────────────────────────────────────────
#
# The frame is tilted slightly. Compute the tilt angle from its top edge.
# Then rotate the scaled painting to match, and translate its center to
# the frame's center.

dx = frame_tr[0] - frame_tl[0]
dy = frame_tr[1] - frame_tl[1]
frame_angle_deg = math.degrees(math.atan2(dy, dx))  # tilt of frame's top edge

frame_center = frame_corners.mean(axis=0)   # frame centroid

theta = math.radians(frame_angle_deg)
cos_t = math.cos(theta)
sin_t = math.sin(theta)

# Rotation around scaled painting center
cx_s, cy_s = scaled_w / 2.0, scaled_h / 2.0

T1 = np.array([[1, 0, -cx_s], [0, 1, -cy_s], [0, 0, 1]], dtype=np.float64)
R  = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
T2 = np.array([[1, 0, frame_center[0]], [0, 1, frame_center[1]], [0, 0, 1]], dtype=np.float64)

M_rigid_3x3 = T2 @ R @ T1

print(f"\nStep 2 - Rigid Transformation (3x3 matrix):")
print(f"  Frame tilt angle: {frame_angle_deg:.2f} deg")
print(f"  Frame center:     ({frame_center[0]:.0f}, {frame_center[1]:.0f})")
print(f"  M_rigid =")
print(np.round(M_rigid_3x3, 4))

M_rigid_2x3 = M_rigid_3x3[:2, :]
painting_rigid = cv2.warpAffine(painting_scaled, M_rigid_2x3, (wW, wH),
                                 flags=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Projective Transformation - warp 4 corners into frame corners
# ─────────────────────────────────────────────────────────────────────────────
#
# After the rigid step the painting is roughly in position but not
# perspective-warped to fit the angled frame.
# We pick the 4 corners of the scaled painting (in the warp output),
# map them to the frame's 4 corners, and compute a Homography H.
#
# src_pts: 4 corners of the painting (after rigid transform)
# dst_pts: 4 corners of the frame (target)

# The 4 corners of the scaled painting in its own coordinate frame
painting_corners_local = np.float32([
    [0,        0       ],   # top-left
    [scaled_w, 0       ],   # top-right
    [scaled_w, scaled_h],   # bottom-right
    [0,        scaled_h],   # bottom-left
])

# Apply the rigid transform to find where those corners land in the wide canvas
def apply_affine_pts(M_2x3, pts):
    """Apply a 2x3 affine matrix to Nx2 points."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M_2x3 @ pts_h.T).T

src_pts_proj = apply_affine_pts(M_rigid_2x3, painting_corners_local).astype(np.float32)

# Target: the frame's actual 4 corners (with a small inset to avoid covering frame border)
inset = 9  # px inset to stay inside the gold border
frame_inset = np.float32([
    frame_tl + np.array([ inset,  inset]),
    frame_tr + np.array([-inset,  inset]),
    frame_br + np.array([-inset, -inset]),
    frame_bl + np.array([ inset, -inset]),
])

H_proj, _ = cv2.findHomography(src_pts_proj, frame_inset)

print(f"\nStep 3 - Projective Transformation (Homography 3x3):")
print("  Painting corners (after rigid):")
for i, pt in enumerate(src_pts_proj):
    print(f"    {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
print("  Frame corners (target):")
for i, pt in enumerate(frame_inset):
    print(f"    {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
print(f"  H =")
print(np.round(H_proj, 6))


# ─────────────────────────────────────────────────────────────────────────────
# Apply the full projective warp to the original scaled painting
# ─────────────────────────────────────────────────────────────────────────────
#
# Rather than warping twice, compose all transforms:
# H_total = H_proj @ M_rigid_augmented @ M_scale_augmented
# But it's cleaner to warpPerspective on the scaled painting directly
# using the composed homography.

# Augment M_rigid_3x3 for composition
# H_proj maps "canvas after rigid" -> frame
# M_rigid_3x3 maps "painting_scaled coords" -> "canvas coords"
# So: H_total maps painting_scaled -> frame

H_total = H_proj @ M_rigid_3x3

print(f"\nComposed Full Transform H_total = H_proj @ M_rigid:")
print(np.round(H_total, 6))

# Warp the scaled painting using the composed homography
painting_final = cv2.warpPerspective(painting_scaled, H_total, (wW, wH),
                                     flags=cv2.INTER_LINEAR)

# Composite: insert painting onto wall (only where painting is non-black)
result = museum_wall.copy()
mask = painting_final.sum(axis=2) > 0
result[mask] = painting_final[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Task 10: The Master Forger - Full Transformation Hierarchy',
             fontsize=15, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(painting, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'Original Painting\n({pW}x{pH})')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(painting_scaled, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'Step 1: Linear Scale\n({scaled_w}x{scaled_h}, s={s_uniform:.3f})')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(museum_wall, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Museum Wall\n(Empty angled frame)')
axes[0, 2].axis('off')

# Show rigid result (small painting placed on canvas)
wall_with_rigid = museum_wall.copy()
mask_r = painting_rigid.sum(axis=2) > 0
wall_with_rigid[mask_r] = painting_rigid[mask_r]
axes[1, 0].imshow(cv2.cvtColor(wall_with_rigid, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Step 2: Rigid Transform\n(Rotate + Translate near frame)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Step 3: Projective Warp\n(4-corner fit into frame)')
axes[1, 1].axis('off')

# Final zoomed view
frame_x_min = int(min(frame_corners[:, 0])) - 20
frame_x_max = int(max(frame_corners[:, 0])) + 20
frame_y_min = int(min(frame_corners[:, 1])) - 20
frame_y_max = int(max(frame_corners[:, 1])) + 20
zoomed = result[frame_y_min:frame_y_max, frame_x_min:frame_x_max]
axes[1, 2].imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Final Result (Zoomed)\nPainting fitted in frame')
axes[1, 2].axis('off')

fig.text(0.5, 0.01,
         "Pipeline: 2x2 Scale  ->  3x3 Rigid (Rotate+Translate)  ->  3x3 Projective Homography",
         ha='center', fontsize=10, color='steelblue')

plt.tight_layout()
plt.savefig('output/task10_master_forger.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task10_master_forger.png")

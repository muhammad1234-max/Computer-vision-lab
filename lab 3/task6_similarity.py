"""
Task 6: The Architect's Blueprint (Similarity Transformation)

Scenario:
    You imported a house blueprint into your software, but it is too small,
    rotated randomly, and sitting in the wrong corner.

Task:
    - A Similarity Transformation preserves angles but changes size.
    - Combine Scale, Rotation, and Translation into one single 3x3 matrix.
    - Apply the single 3x3 operation to fix the blueprint in one mathematical step.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.makedirs('output', exist_ok=True)


def make_blueprint(width=200, height=150):
    """
    Creates a simple architectural blueprint (floor plan) image.
    Blue background with white lines representing walls and rooms.
    """
    img = np.full((height, width, 3), (30, 30, 80), dtype=np.uint8)  # dark blue bg

    # Outer walls
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (200, 220, 255), 2)

    # Interior walls (rooms)
    mid_x, mid_y = width // 2, height // 2
    cv2.line(img, (mid_x, 10), (mid_x, height - 10), (180, 200, 240), 1)
    cv2.line(img, (10, mid_y), (width - 10, mid_y), (180, 200, 240), 1)

    # Door openings (gaps in walls)
    cv2.line(img, (mid_x - 15, mid_y), (mid_x + 15, mid_y), (30, 30, 80), 2)
    cv2.line(img, (mid_x, mid_y - 10), (mid_x, mid_y + 10), (30, 30, 80), 2)

    # Room labels
    cv2.putText(img, "LR", (mid_x // 2 - 10, mid_y // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 255), 1)
    cv2.putText(img, "BR", (mid_x + mid_x // 2 - 10, mid_y // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 255), 1)
    cv2.putText(img, "KT", (mid_x // 2 - 10, mid_y + mid_y // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 255), 1)
    cv2.putText(img, "BA", (mid_x + mid_x // 2 - 10, mid_y + mid_y // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 255), 1)

    # North arrow
    cv2.arrowedLine(img, (width - 25, height - 15), (width - 25, height - 35),
                    (0, 255, 200), 2, tipLength=0.4)
    cv2.putText(img, "N", (width - 29, height - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 200), 1)

    return img


# Create the blueprint and simulate the "bad" import
blueprint_clean = make_blueprint(200, 150)
canvas_size = 600
bh, bw = blueprint_clean.shape[:2]

# Simulate wrong state: tiny (scale 0.5), rotated 30, placed at corner (20, 20)
bad_scale  = 0.5
bad_angle  = 30.0
bad_pos    = (20, 20)

M_bad = cv2.getRotationMatrix2D((bw / 2, bh / 2), bad_angle, bad_scale)
M_bad[0, 2] += bad_pos[0] - bw * bad_scale / 2
M_bad[1, 2] += bad_pos[1] - bh * bad_scale / 2

canvas_bad = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
cv2.warpAffine(blueprint_clean, M_bad, (canvas_size, canvas_size),
               dst=canvas_bad, flags=cv2.INTER_LINEAR)

print("=" * 50)
print("Task 6: The Architect's Blueprint - Similarity Transformation")
print("=" * 50)
print(f"\nBlueprint is: scale={bad_scale}x, rotated={bad_angle} deg, at corner {bad_pos}.")
print("Goal: scale up 2x, rotate -30 deg to align, and move to center.\n")


# Step 1: Define the desired parameters
target_scale  = 2.0
target_angle  = 0.0
target_cx     = canvas_size / 2.0
target_cy     = canvas_size / 2.0

correct_scale = target_scale / bad_scale   # 2.0 / 0.5 = 4.0
correct_angle = -(bad_angle - target_angle)  # -30 deg

print(f"Correction needed: scale by {correct_scale:.1f}, rotate {correct_angle} deg,")
print(f"  then center at ({target_cx:.0f}, {target_cy:.0f}) on canvas.\n")


# Step 2: Build the Similarity Transformation matrix (3x3)
#
# A Similarity matrix has the form:
#
#   M = s * [ cos(t)  -sin(t)  tx ]
#           [ sin(t)   cos(t)  ty ]
#           [   0        0      1 ]
#
# It combines uniform scale (s), rotation (t), and translation (tx, ty).
# Angles and shapes are preserved; only scale and position change.

theta = math.radians(correct_angle)
cos_t = math.cos(theta)
sin_t = math.sin(theta)
s     = correct_scale

cx_bp, cy_bp = bw / 2.0, bh / 2.0

T_to_origin = np.array([
    [1, 0, -cx_bp],
    [0, 1, -cy_bp],
    [0, 0, 1     ]
], dtype=np.float64)

SR = np.array([
    [s * cos_t, -s * sin_t, 0],
    [s * sin_t,  s * cos_t, 0],
    [0,          0,         1]
], dtype=np.float64)

T_to_canvas = np.array([
    [1, 0, target_cx],
    [0, 1, target_cy],
    [0, 0, 1        ]
], dtype=np.float64)

M_similarity_3x3 = T_to_canvas @ SR @ T_to_origin

print("3x3 Similarity Transformation Matrix (Scale + Rotation + Translation):")
print(np.round(M_similarity_3x3, 4))
print(f"\n  Uniform scale s = {s}")
print(f"  Rotation angle  = {correct_angle} deg")
print(f"  Final center    = ({target_cx:.0f}, {target_cy:.0f})")


# Step 3: Apply the single matrix
M_2x3 = M_similarity_3x3[:2, :]

canvas_fixed = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
cv2.warpAffine(blueprint_clean, M_2x3, (canvas_size, canvas_size),
               dst=canvas_fixed, flags=cv2.INTER_LINEAR)


# Visualize
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Task 6: The Architect's Blueprint - Similarity Transformation",
             fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(blueprint_clean, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Blueprint\n(200x150, proper orientation)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(canvas_bad, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Bad Import\n(scale={bad_scale}, angle={bad_angle} deg, wrong corner)')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(canvas_fixed, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Fixed Blueprint\n(scale=2x, 0 deg, centered - ONE 3x3 matrix)')
axes[2].axis('off')

fig.text(0.5, 0.01,
         f"Similarity M = s*[[cos(t),-sin(t),tx],[sin(t),cos(t),ty]]  |  s={s}, t={correct_angle} deg",
         ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('output/task6_similarity.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task6_similarity.png")

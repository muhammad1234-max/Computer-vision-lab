"""
Task 5: The Robot's Assembly Line (Rigid Transformation)

Scenario:
    A robotic arm needs to pick up a microchip on a conveyor belt.
    The chip is rotated sideways and slightly off-center.

Task:
    - A Rigid Transformation preserves exact shape of an object
    - Combine Rotation and Translation matrices into a SINGLE 3x3 matrix
    - Apply it simultaneously to align the chip with the robot's gripper
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.makedirs('output', exist_ok=True)


def make_chip_image(size=80):
    """
    Creates a simple microchip image - a square with pins on the sides.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 50   # dark background

    chip_w, chip_h = 50, 40
    cx, cy = size // 2, size // 2

    # Chip body (dark green)
    x1, y1 = cx - chip_w // 2, cy - chip_h // 2
    x2, y2 = cx + chip_w // 2, cy + chip_h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 80, 20), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 160, 0), 2)

    # Pins on top and bottom
    pin_spacing = 8
    for i in range(4):
        px = cx - 16 + i * pin_spacing
        cv2.rectangle(img, (px, y1 - 6), (px + 4, y1), (180, 180, 180), -1)
        cv2.rectangle(img, (px, y2), (px + 4, y2 + 6), (180, 180, 180), -1)

    # Chip markings
    cv2.putText(img, "IC", (cx - 9, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    cv2.circle(img, (x1 + 5, y1 + 5), 3, (200, 200, 0), -1)  # orientation dot

    return img


# Create a chip image at some known position (center of a 300x300 canvas)
canvas_size = 300
chip_img = make_chip_image(80)
chip_h, chip_w = chip_img.shape[:2]

# Place chip at origin area (we'll transform it to the target)
canvas_before = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 30
gripper_pos = (220, 160)   # where the robot gripper is waiting

# Chip is currently rotated 90 degrees and sitting at (50, 50)
chip_start_angle = 90.0          # chip is turned sideways
chip_start_pos = (50, 50)        # chip's current center position
target_pos = gripper_pos         # where it needs to go (gripper location)
target_angle = 0.0               # gripper expects chip upright

# Simulate the current (wrong) state: chip rotated 90 deg, off position
M_sim = cv2.getRotationMatrix2D((chip_w / 2, chip_h / 2), chip_start_angle, 1.0)
M_sim[0, 2] += chip_start_pos[0] - chip_w // 2
M_sim[1, 2] += chip_start_pos[1] - chip_h // 2
canvas_before_state = canvas_before.copy()
cv2.warpAffine(chip_img, M_sim, (canvas_size, canvas_size),
               dst=canvas_before_state, flags=cv2.INTER_LINEAR)

# Mark the gripper position
cv2.drawMarker(canvas_before_state, gripper_pos, (0, 0, 255),
               cv2.MARKER_CROSS, 20, 2)
cv2.putText(canvas_before_state, "Gripper", (gripper_pos[0] - 30, gripper_pos[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


print("=" * 50)
print("Task 5: Robot's Assembly Line - Rigid Transformation")
print("=" * 50)
print(f"\nChip is at position {chip_start_pos}, rotated {chip_start_angle} degrees.")
print(f"Gripper is at {target_pos}, expecting chip at 0 degrees rotation.\n")


# --- Step 1: Build the Rotation Matrix (3x3) ---
#
# We need to rotate the chip by -90 degrees (from 90 back to 0)
# A rigid rotation in 3x3 homogeneous form:
#
#   R_3x3 = [ cos(t)  -sin(t)  0 ]
#            [ sin(t)   cos(t)  0 ]
#            [   0        0     1 ]

rotation_angle = -(chip_start_angle - target_angle)  # -90 degrees to correct
theta = math.radians(rotation_angle)
cos_t = math.cos(theta)
sin_t = math.sin(theta)

R_3x3 = np.array([
    [cos_t, -sin_t, 0.0],
    [sin_t,  cos_t, 0.0],
    [0.0,    0.0,   1.0]
], dtype=np.float64)

print("3x3 Rotation Matrix (rotate -90 degrees):")
print(np.round(R_3x3, 4))


# --- Step 2: Build the Translation Matrix (3x3) ---
#
# Move the chip's CENTER from chip_start_pos to target_pos:
#
#   T_3x3 = [ 1  0  tx ]
#            [ 0  1  ty ]
#            [ 0  0   1 ]

tx = target_pos[0] - chip_start_pos[0]
ty = target_pos[1] - chip_start_pos[1]

T_3x3 = np.array([
    [1.0, 0.0, tx],
    [0.0, 1.0, ty],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

print(f"\n3x3 Translation Matrix (move by dx={tx}, dy={ty}):")
print(T_3x3)


# --- Step 3: Combine into ONE Rigid Transformation Matrix ---
#
# A Rigid transformation = Rotation + Translation (no scaling, no shear).
# We multiply them into a single 3x3 matrix:
#
#   M_rigid = T @ R    (first rotate, then translate)
#
# The ORDER matters! We rotate first (around origin), then translate.
# Doing it in one step is much cleaner than applying two separate transforms.

# We also need to account for rotating AROUND the chip's center, not (0,0).
# So we: translate chip center to origin -> rotate -> translate back -> final move

# Full rigid transform around chip center then to target:
cx_chip = chip_start_pos[0]
cy_chip = chip_start_pos[1]

# Build it step by step (all in 3x3):
# T1: move chip center to origin
T1 = np.array([[1, 0, -cx_chip], [0, 1, -cy_chip], [0, 0, 1]], dtype=np.float64)
# R: rotate
R = R_3x3
# T2: move to target position
T2 = np.array([[1, 0, target_pos[0]], [0, 1, target_pos[1]], [0, 0, 1]], dtype=np.float64)

# Combine: first T1, then R, then T2
M_rigid_3x3 = T2 @ R @ T1

print("\nSingle Combined Rigid Transformation Matrix (T @ R):")
print(np.round(M_rigid_3x3, 4))
print("\nThis single matrix does: rotate -90° AND move to gripper in one operation!")


# --- Step 4: Apply the rigid transformation ---
# Extract the 2x3 portion for warpAffine
M_2x3 = M_rigid_3x3[:2, :]

canvas_after = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 30

# Warp the chip
cv2.warpAffine(chip_img, M_2x3, (canvas_size, canvas_size),
               dst=canvas_after, flags=cv2.INTER_LINEAR)

# Draw gripper marker on result
cv2.drawMarker(canvas_after, gripper_pos, (0, 255, 0),
               cv2.MARKER_CROSS, 20, 2)
cv2.putText(canvas_after, "Gripper", (gripper_pos[0] - 30, gripper_pos[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)


# --- Visualize ---
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Task 5: Robot Assembly Line - Rigid Transformation',
             fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(canvas_before_state, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Before: chip at {chip_start_pos}, rotated {chip_start_angle}°\nRed cross = gripper target')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(canvas_after, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'After: chip moved to gripper {target_pos}, aligned 0°\n(Single 3x3 Rigid matrix applied)')
axes[1].axis('off')

fig.text(0.5, 0.01,
         "Rigid = Rotation + Translation in one 3x3 matrix  |  Shape preserved (no scaling or shear)",
         ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('output/task5_rigid.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n[Done] Output saved to output/task5_rigid.png")

"""
Task 3: The Fast Train Barcode (Linear - Shear)

Scenario:
    A scanner took a photo of a barcode on a fast-moving train,
    causing the image to look slanted to the right.

Task:
    - Construct a 2x2 shear matrix to "push" pixels horizontally
    - Shift the slanted lines back into a perfect rectangle
    - The barcode should be readable again
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)


def make_barcode_image(width=300, height=120):
    """
    Creates a simple black-and-white barcode image.
    Bar widths vary to simulate a real barcode pattern.
    """
    img = np.ones((height, width), dtype=np.uint8) * 255  # white background

    # Draw vertical black bars of varying widths at different positions
    # This simulates a typical 1D barcode pattern
    bar_positions = [
        (10, 14), (20, 22), (30, 34), (40, 41), (50, 54),
        (60, 64), (70, 75), (80, 82), (90, 95), (100, 105),
        (115, 117), (125, 130), (140, 146), (155, 157),
        (165, 172), (180, 183), (190, 197), (205, 208),
        (215, 222), (235, 240), (250, 256), (265, 268),
        (275, 282), (288, 295)
    ]

    for x_start, x_end in bar_positions:
        if x_end < width:
            cv2.rectangle(img, (x_start, 10), (x_end, height - 20), 0, -1)

    # Add a thin quiet zone line at top and bottom
    cv2.line(img, (5, 8), (width - 5, 8), 100, 1)
    cv2.line(img, (5, height - 18), (width - 5, height - 18), 100, 1)

    # Add some fake digits below the barcode
    cv2.putText(img, "4 710085 252218", (15, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)

    return img


# Create a clean barcode, then apply shear to simulate the train motion blur effect
barcode_clean = make_barcode_image()
h, w = barcode_clean.shape[:2]

# The shear amount - how much the train motion tilted the scan
# A shear of 0.4 means pixels shift right by 0.4 * their y-coordinate
shear_amount = 0.4

# Forward shear matrix (what happened to the barcode during scanning):
#
#   [ x' ]   [ 1  shx ] [ x ]       x' = x + shx * y
#   [ y' ] = [ 0   1  ] [ y ]       y' = y  (rows don't move)
#
# This slides each row horizontally by shear_amount * row_number

M_forward = np.float32([
    [1, shear_amount, 0],
    [0, 1,            0]
])

# The shear also makes the image wider, so we need a bigger canvas
extra_width = int(h * shear_amount)
sheared_w = w + extra_width

barcode_sheared = cv2.warpAffine(
    barcode_clean, M_forward, (sheared_w, h),
    borderValue=255
)

print("=" * 50)
print("Task 3: The Fast Train Barcode - Linear Shear")
print("=" * 50)
print(f"\nBarcode original size: {w} x {h}")
print(f"Shear amount: {shear_amount} (each pixel shifts right by {shear_amount} * its y position)")
print(f"Sheared canvas size: {sheared_w} x {h}\n")


# --- Step 1: Construct the 2x2 Correction Shear Matrix ---
#
# To UNDO a shear of +shx, we apply a shear of -shx:
#
#   M_inv = [ 1  -shx ]   ->   x_corrected = x - shx * y
#            [ 0    1  ]        y_corrected = y
#
# This pushes each row back to the LEFT by the same amount it was pushed right.

shx_correct = -shear_amount   # negative to invert the shear

S_inv = np.array([
    [1, shx_correct],
    [0, 1          ]
], dtype=np.float64)

print("2x2 Correction Shear Matrix:")
print(S_inv)
print(f"\nThis shifts each pixel LEFT by {abs(shx_correct)} * its y-coordinate")
print("Effectively un-tilting the slanted bars back to vertical.\n")


# --- Step 2: Build full 2x3 matrix with shear correction ---
#
# We also need a small horizontal offset to prevent negative x values.
# Since the correction shear will move some pixels to the left, we
# need to ensure nothing gets clipped off the left edge.

# Offset to keep the corrected image inside the canvas
# The topmost row (y=0) doesn't shift, but bottom row (y=h) shifts by shx_correct * h
# If shx_correct is negative, top-right pixels shift left - we need padding on left
tx_offset = 0 if shx_correct >= 0 else int(abs(shx_correct) * h)

M_correct = np.float32([
    [1, shx_correct, tx_offset],
    [0, 1,           0        ]
])

print("2x3 Shear Correction Matrix (with edge offset):")
print(M_correct)


# --- Step 3: Apply the correction ---
corrected = cv2.warpAffine(
    barcode_sheared, M_correct,
    (sheared_w, h),
    flags=cv2.INTER_LINEAR,
    borderValue=255   # white background
)


# --- Visualize ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Task 3: The Fast Train Barcode - Linear Shear Correction',
             fontsize=14, fontweight='bold')

axes[0].imshow(barcode_clean, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original Barcode\n(perfectly vertical bars)')
axes[0].axis('off')

axes[1].imshow(barcode_sheared, cmap='gray', vmin=0, vmax=255)
axes[1].set_title(f'Scanned on Moving Train\n(shear={shear_amount}, bars slanted right)')
axes[1].axis('off')

axes[2].imshow(corrected, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Corrected Barcode\n(bars straightened, readable again)')
axes[2].axis('off')

fig.text(0.5, 0.01,
         f"Shear matrix: [[1, {shx_correct}], [0, 1]]  "
         f"| Applied inverse shear to un-tilt the bars",
         ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('output/task3_shear.png', dpi=100, bbox_inches='tight')
plt.show()

print("[Done] Output saved to output/task3_shear.png")

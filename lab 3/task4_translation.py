"""
Task 4: The Hidden Treasure Map (Affine - Translation)

Scenario:
    A digital map is misaligned - the "X marks the spot" is hidden off screen,
    roughly 150 pixels to the left and 80 pixels up from where it should be.

Task:
    - Upgrade from 2x2 to 3x3 matrices to allow translation
    - Create a translation matrix to shift every pixel 150 pixels right and 80 pixels up
    - Slide the map back into the visible frame
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)


def make_treasure_map(width=500, height=400):
    """
    Creates a synthetic treasure map image with an 'X marks the spot'.
    The X will be placed near the top-left so after -150, -80 it goes off screen.
    """
    # Parchment-like background color (warm yellowish-tan)
    img = np.full((height, width, 3), (190, 210, 230), dtype=np.uint8)

    # Add some texture with random noise to look like old paper
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw coastline / island outline
    island_pts = np.array([
        [100, 100], [200, 80], [320, 90], [400, 150],
        [420, 250], [350, 330], [200, 350], [100, 280], [80, 180]
    ], dtype=np.int32)
    cv2.polylines(img, [island_pts], isClosed=True, color=(80, 60, 40), thickness=3)
    cv2.fillPoly(img, [island_pts], color=(160, 190, 140))  # greenish land

    # Some landmarks (trees, mountain)
    cv2.circle(img, (160, 200), 12, (50, 120, 50), -1)   # tree
    cv2.circle(img, (300, 180), 10, (50, 120, 50), -1)   # tree
    cv2.circle(img, (250, 150), 15, (120, 100, 80), -1)  # mountain

    # Draw a dotted path to the treasure
    path_pts = [(160, 200), (220, 190), (250, 160), (270, 200), (290, 220)]
    for i in range(len(path_pts) - 1):
        cv2.line(img, path_pts[i], path_pts[i+1], (80, 40, 10), 1,
                 lineType=cv2.LINE_AA)

    # The TREASURE - big red X!
    tx, ty = 295, 225   # treasure location
    size = 18
    cv2.line(img, (tx - size, ty - size), (tx + size, ty + size), (0, 0, 200), 4)
    cv2.line(img, (tx + size, ty - size), (tx - size, ty + size), (0, 0, 200), 4)
    cv2.circle(img, (tx, ty), 5, (0, 0, 255), -1)

    # Label
    cv2.putText(img, "X", (tx - 8, ty + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
    cv2.putText(img, "Treasure!", (tx + 20, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 20, 20), 1)

    # Map border
    cv2.rectangle(img, (5, 5), (width - 5, height - 5), (80, 60, 40), 3)

    return img


# Create the map, then shift it so the X disappears off-screen
map_correct = make_treasure_map(500, 400)
h, w = map_correct.shape[:2]

# The misalignment: map needs to move RIGHT 150 and UP 80 to be correct.
# Which means currently it's shifted LEFT 150 and DOWN 80 from correct position.
# We'll simulate the "wrong" position by applying the inverse shift.
dx_needed = 150    # need to move right 150 px
dy_needed = -80    # need to move up 80 px (negative y = up in image coords)

# Apply the WRONG shift to create the misaligned version
M_wrong = np.float32([
    [1, 0, -dx_needed],   # shift left 150 (hiding the X off-screen)
    [0, 1, -dy_needed]    # shift down 80
])

canvas_w, canvas_h = w, h
map_misaligned = cv2.warpAffine(map_correct, M_wrong, (canvas_w, canvas_h),
                                borderValue=(120, 120, 100))


print("=" * 50)
print("Task 4: The Hidden Treasure Map - Affine Translation")
print("=" * 50)
print("\nThe map is misaligned! The X is off-screen.")
print(f"We need to shift it: +{dx_needed}px right, +{abs(dy_needed)}px up\n")


# --- Step 1: Upgrade to 3x3 Matrix for Translation ---
#
# A 2x2 linear matrix CANNOT do translation.
# By adding a third row/column (homogeneous coordinates), we can:
#
#   [ x' ]   [ 1  0  tx ] [ x ]
#   [ y' ] = [ 0  1  ty ] [ y ]
#   [ 1  ]   [ 0  0   1 ] [ 1 ]
#
# This is now an AFFINE transformation. The 1 in the bottom-right keeps
# the homogeneous coordinate stable during the operation.

tx = float(dx_needed)    # translate right by 150 pixels
ty = float(dy_needed)    # translate up by 80 pixels (negative y)

T_3x3 = np.array([
    [1.0, 0.0, tx],
    [0.0, 1.0, ty],
    [0.0, 0.0, 1.0]   # this row makes it a proper 3x3 homogeneous matrix
], dtype=np.float64)

print("3x3 Translation Matrix T:")
print(T_3x3)
print(f"\ntx = +{tx} (shift RIGHT)")
print(f"ty = {ty} (shift UP, negative because image y increases downward)")
print("\nThe bottom row [0, 0, 1] is the homogeneous coordinate row.")
print("Without it we'd have just a 2D linear matrix that can't translate.\n")


# --- Step 2: Extract 2x3 for warpAffine ---
#
# cv2.warpAffine() takes the top 2 rows of the 3x3 matrix.
# The last row [0,0,1] is implied.

M_translate = T_3x3[:2, :]   # grab just the first 2 rows

print("2x3 Matrix passed to warpAffine (top 2 rows of T_3x3):")
print(M_translate)


# --- Step 3: Apply the correction ---
corrected = cv2.warpAffine(
    map_misaligned, M_translate,
    (canvas_w, canvas_h),
    flags=cv2.INTER_LINEAR,
    borderValue=(120, 120, 100)
)


# --- Visualize ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Task 4: The Hidden Treasure Map - Affine Translation',
             fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(map_correct, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Map\n(X visible)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(map_misaligned, cv2.COLOR_BGR2RGB))
axes[1].set_title('Misaligned Map\n(X is off-screen!)')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Corrected Map\n(shifted +{dx_needed}px right, +{abs(dy_needed)}px up)')
axes[2].axis('off')

fig.text(0.5, 0.01,
         f"3x3 Translation Matrix T = [[1, 0, {tx:.0f}], [0, 1, {ty:.0f}], [0, 0, 1]]",
         ha='center', fontsize=10, color='steelblue')

plt.tight_layout()
plt.savefig('output/task4_translation.png', dpi=100, bbox_inches='tight')
plt.show()

print("[Done] Output saved to output/task4_translation.png")

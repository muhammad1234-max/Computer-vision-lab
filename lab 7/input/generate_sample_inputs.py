"""
Generate realistic sample images for testing the Computer Vision lab tasks.
Creates:
- input/sample_image.png (for Q1a, Q1c, Q1d)
- input/tilted_book.png (for Q1e)
- input/document_tilted.png (for Q2)
- input/checkerboard.png (for Q3)
- input/colmap_images/image_01.png .. image_06.png (for Q4)
"""

import os
import cv2
import numpy as np


def generate_sample_grid_image(path):
    """Generate a clean test image with grid lines, shapes, and text."""
    h, w = 500, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 245

    # Draw grid lines to easily track straightness and deformation
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w, y), (210, 210, 210), 1)
    for x in range(0, w, 50):
        cv2.line(img, (x, 0), (x, h), (210, 210, 210), 1)

    # Outer border
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (40, 40, 40), 3)

    # Geometric shapes
    cv2.rectangle(img, (60, 60), (200, 200), (45, 120, 210), -1)  # Blue rectangle
    cv2.circle(img, (350, 140), 65, (40, 180, 70), -1)            # Green circle
    pts = np.array([[250, 240], [170, 380], [330, 380]], np.int32)
    cv2.fillPoly(img, [pts], (210, 80, 45))                        # Red triangle

    # Diagonal straight reference line across image
    cv2.line(img, (30, 450), (470, 50), (180, 50, 180), 2)

    # Informative text
    cv2.putText(img, "Computer Vision Lab 07", (60, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "Reference Grid & Shapes", (120, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)

    cv2.imwrite(path, img)
    print(f"Saved: {path}")


def generate_tilted_surface_image(path, title="COMPUTER VISION"):
    """
    Generate an image simulating a tilted book or document resting on a desk.
    Returns the true corner coordinates in the tilted image.
    """
    # Create front-facing flat document first
    doc_w, doc_h = 600, 400
    doc = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 255

    # Document border & header
    cv2.rectangle(doc, (15, 15), (doc_w - 15, doc_h - 15), (20, 20, 80), 3)
    cv2.rectangle(doc, (25, 25), (doc_w - 25, 100), (230, 220, 200), -1)
    cv2.putText(doc, title, (50, 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 120), 2)

    # Document text lines
    lines = [
        "Lab 07: Perspective Correction and Homography",
        "Source: Perspective-distorted surface",
        "Destination: Front-facing unwarped rectangle",
        "Preserves collinearity along any projective line.",
        "National University of Computer & Emerging Sciences"
    ]
    for i, line in enumerate(lines):
        cv2.putText(doc, line, (40, 150 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    # Draw a barcode / stamp
    cv2.rectangle(doc, (420, 270), (560, 360), (220, 240, 255), -1)
    cv2.rectangle(doc, (420, 270), (560, 360), (30, 80, 150), 2)
    cv2.putText(doc, "FAST NUCES", (435, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 80, 150), 1)

    # Create background scene (desk)
    scene_w, scene_h = 900, 700
    scene = np.ones((scene_h, scene_w, 3), dtype=np.uint8)
    # Wood grain tone
    scene[:] = [140, 170, 205]
    for y in range(0, scene_h, 8):
        shade = int(140 + 15 * np.sin(y / 15.0))
        scene[y:y+4, :] = [shade - 20, shade + 10, shade + 45]

    # Warp flat document into tilted perspective in scene
    # Source corners (flat document)
    src_corners = np.float32([
        [0, 0],
        [doc_w - 1, 0],
        [doc_w - 1, doc_h - 1],
        [0, doc_h - 1]
    ])

    # Tilted target corners in scene (perspective foreshortening)
    # Top edge is further away (shorter), bottom is closer (wider)
    tilted_corners = np.float32([
        [220, 120],  # Top-left
        [680, 160],  # Top-right
        [780, 560],  # Bottom-right
        [120, 500]   # Bottom-left
    ])

    H = cv2.getPerspectiveTransform(src_corners, tilted_corners)
    warped_doc = cv2.warpPerspective(doc, H, (scene_w, scene_h))

    # Mask to blend warped document onto background
    mask = np.zeros((scene_h, scene_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tilted_corners), 255)

    # Add subtle shadow under tilted document
    shadow_mask = np.zeros((scene_h, scene_w), dtype=np.uint8)
    shadow_corners = tilted_corners + np.float32([15, 15])
    cv2.fillConvexPoly(shadow_mask, np.int32(shadow_corners), 100)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 0)
    scene[shadow_mask > 0] = (scene[shadow_mask > 0] * 0.7).astype(np.uint8)

    scene[mask > 0] = warped_doc[mask > 0]

    cv2.imwrite(path, scene)
    print(f"Saved: {path}")
    return tilted_corners


def generate_checkerboard(path, cols=9, rows=6, square_size=60):
    """Generate a standard checkerboard calibration pattern."""
    img_w = (cols + 1) * square_size
    img_h = (rows + 1) * square_size
    board = np.ones((img_h, img_w), dtype=np.uint8) * 255

    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 1:
                y1 = r * square_size
                y2 = (r + 1) * square_size
                x1 = c * square_size
                x2 = (c + 1) * square_size
                board[y1:y2, x1:x2] = 0

    # Add white margin
    margin = 50
    final = np.ones((img_h + 2 * margin, img_w + 2 * margin), dtype=np.uint8) * 255
    final[margin:margin + img_h, margin:margin + img_w] = board

    cv2.imwrite(path, final)
    print(f"Saved: {path}")


def generate_colmap_multiview_images(output_dir, num_views=8):
    """
    Generate textured 3D multi-view images with high-frequency SIFT keypoints
    suitable for feature matching, camera pose estimation, sparse point cloud,
    and 3D mesh reconstruction.
    """
    os.makedirs(output_dir, exist_ok=True)

    tex = cv2.imread("input/sample_image.png")
    if tex is None:
        generate_sample_grid_image("input/sample_image.png")
        tex = cv2.imread("input/sample_image.png")
    th, tw = tex.shape[:2]

    # Overlay textured coordinate grid to ensure dense feature descriptors
    for y in range(0, th, 20):
        for x in range(0, tw, 20):
            cv2.putText(tex, f"{x},{y}", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    corners_3d = np.array([
        [-1.0, -0.7, 0.0],
        [ 1.0, -0.7, 0.0],
        [ 1.0,  0.7, 0.0],
        [-1.0,  0.7, 0.0]
    ], dtype=np.float32)

    src_uv = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    f = 600.0
    cx, cy = 400.0, 300.0

    angles = np.linspace(-18, 18, num_views)
    for i, deg in enumerate(angles):
        rad = np.radians(deg)
        cam_pos = np.array([3.0 * np.sin(rad), 0.2 * np.sin(rad * 2), -3.0 * np.cos(rad)], dtype=np.float32)
        fwd = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        R_wc = np.vstack([right, up, fwd])
        t_wc = -R_wc @ cam_pos

        pts_cam = (R_wc @ corners_3d.T).T + t_wc
        u = f * pts_cam[:, 0] / pts_cam[:, 2] + cx
        v = f * pts_cam[:, 1] / pts_cam[:, 2] + cy
        dst_uv = np.column_stack([u, v]).astype(np.float32)

        H = cv2.getPerspectiveTransform(src_uv, dst_uv)
        bg = np.random.RandomState(i).randint(200, 240, (600, 800, 3), dtype=np.uint8)
        warped = cv2.warpPerspective(tex, H, (800, 600), borderMode=cv2.BORDER_TRANSPARENT)
        mask = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (800, 600))
        bg[mask > 0] = warped[mask > 0]

        img_path = os.path.join(output_dir, f"view_{i+1:02d}.jpg")
        cv2.imwrite(img_path, bg)

    print(f"Generated {num_views} rich multi-view images in {output_dir}")


if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    generate_sample_grid_image("input/sample_image.png")
    generate_tilted_surface_image("input/tilted_book.png", "COMPUTER VISION TEXTBOOK")
    generate_tilted_surface_image("input/document_tilted.png", "EXAMINATION DOCUMENT")
    generate_checkerboard("input/checkerboard.png", cols=9, rows=6)
    generate_colmap_multiview_images("input/colmap_images", num_views=6)
    print("All sample input data prepared successfully.")

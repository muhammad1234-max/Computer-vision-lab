"""
selective_search.py
===================
Region proposal generation using OpenCV Selective Search and IoU utilities.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import cv2
import numpy as np


def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes format: [xmin, ymin, xmax, ymax]
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def generate_proposals(image, mode='fast', max_proposals=500, scale=None):
    """
    Generates region proposals using OpenCV Selective Search Segmentation.

    Args:
        image: RGB or BGR numpy image (H, W, 3).
        mode: 'fast' for speed or 'quality' for higher recall.
        max_proposals: Maximum number of proposals to return.
        scale: Optional resize factor to accelerate proposal generation on CPU.

    Returns:
        proposals: List of bounding boxes [xmin, ymin, xmax, ymax].
    """
    orig_h, orig_w = image.shape[:2]

    if scale is not None and scale < 1.0:
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        search_img = cv2.resize(image, (scaled_w, scaled_h))
        fx = orig_w / float(scaled_w)
        fy = orig_h / float(scaled_h)
    else:
        search_img = image
        fx = 1.0
        fy = 1.0

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(search_img)

    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()

    proposals = []
    for i, (x, y, w, h) in enumerate(rects):
        if i >= max_proposals:
            break
        # Map back to original coordinate system
        xmin = int(round(x * fx))
        ymin = int(round(y * fy))
        xmax = int(round((x + w) * fx))
        ymax = int(round((y + h) * fy))

        # Clip to image boundaries
        xmin = max(0, min(orig_w - 1, xmin))
        ymin = max(0, min(orig_h - 1, ymin))
        xmax = max(0, min(orig_w, xmax))
        ymax = max(0, min(orig_h, ymax))

        # Filter out tiny/degenerate boxes
        if (xmax - xmin) >= 10 and (ymax - ymin) >= 10:
            proposals.append([xmin, ymin, xmax, ymax])

    return proposals


if __name__ == "__main__":
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (200, 200), (255, 255, 255), -1)
    props = generate_proposals(test_img, mode='fast', max_proposals=50)
    print(f"Test generated {len(props)} proposals successfully.")
    iou = compute_iou([50, 50, 200, 200], [60, 60, 190, 190])
    print(f"Sample IoU: {iou:.3f}")


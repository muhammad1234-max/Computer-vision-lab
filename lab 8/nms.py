"""
nms.py
======
Non-Maximum Suppression (NMS) algorithm implemented from first principles.
Compares custom implementation against torchvision.ops.nms.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import numpy as np
import torch
import torchvision.ops as ops
from selective_search import compute_iou


def non_max_suppression_np(boxes, scores, iou_threshold=0.3, score_threshold=0.5):
    """
    Applies Non-Maximum Suppression using pure NumPy.

    Args:
        boxes: List or array of shape (N, 4) with [xmin, ymin, xmax, ymax].
        scores: List or array of confidence scores of shape (N,).
        iou_threshold: Overlap threshold above which suppression occurs.
        score_threshold: Minimum confidence score to be considered.

    Returns:
        keep: List of indices of the boxes to keep, sorted by score descending.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Filter out boxes below score threshold
    mask = scores >= score_threshold
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    boxes = boxes[indices]
    scores = scores[indices]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(indices[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(union)
        nonzero = union > 0
        iou[nonzero] = inter[nonzero] / union[nonzero]

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression_torch(boxes, scores, iou_threshold=0.3, score_threshold=0.5):
    """
    Applies Non-Maximum Suppression using torchvision.ops.nms.
    """
    if len(boxes) == 0:
        return []

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)

    mask = scores_tensor >= score_threshold
    valid_indices = torch.where(mask)[0]
    if len(valid_indices) == 0:
        return []

    keep = ops.nms(boxes_tensor[valid_indices], scores_tensor[valid_indices], iou_threshold)
    return valid_indices[keep].cpu().numpy().tolist()


if __name__ == "__main__":
    test_boxes = [
        [100, 100, 200, 200],
        [105, 102, 198, 201],  # Heavy overlap with box 0
        [300, 300, 400, 400],  # Different object
        [110, 108, 195, 195],  # Another overlap
    ]
    test_scores = [0.92, 0.75, 0.88, 0.60]

    kept_np = non_max_suppression_np(test_boxes, test_scores, iou_threshold=0.3, score_threshold=0.5)
    kept_torch = non_max_suppression_torch(test_boxes, test_scores, iou_threshold=0.3, score_threshold=0.5)

    print("Custom NumPy NMS Kept indices:", kept_np)
    print("PyTorch Ops NMS Kept indices: ", kept_torch)
    assert set(kept_np) == set(kept_torch), "NMS implementations do not match!"
    print("NMS verification passed!")


"""
inference.py
============
Inference module for Regional Convolutional Neural Network (R-CNN).
Executes end-to-end detection:
  Selective Search -> Proposal Warping -> CNN Classification & Bounding Box Regression ->
  Regression Offset Decoding -> Confidence Filtering -> Non-Maximum Suppression (NMS) -> Visualization.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import os
import argparse
import cv2
import numpy as np
import torch

from selective_search import generate_proposals
from model import RCNNModel, apply_regression_offsets
from nms import non_max_suppression_np
from dataset import get_rcnn_transforms, parse_voc_xml


def detect_objects_in_image(image_bgr, model, device="cpu",
                            crop_size=128,
                            score_thresh=0.65,
                            nms_thresh=0.3,
                            max_proposals=350):
    """
    Performs full R-CNN inference on a single image.

    Returns:
        detections: List of dicts with 'box', 'score', and 'label'.
        raw_proposals_count: Number of proposals evaluated.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    scale = 0.6 if max(orig_h, orig_w) > 600 else 1.0

    proposals = generate_proposals(image_bgr, mode='fast', max_proposals=max_proposals, scale=scale)
    if len(proposals) == 0:
        return [], 0

    transform = get_rcnn_transforms(crop_size=crop_size, is_train=False)

    crop_tensors = []
    valid_proposals = []

    for p in proposals:
        xmin, ymin, xmax, ymax = p
        crop = image_bgr[ymin:ymax, xmin:xmax]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_t = transform(crop_rgb)
        crop_tensors.append(crop_t)
        valid_proposals.append(p)

    if len(crop_tensors) == 0:
        return [], len(proposals)

    crop_batch = torch.stack(crop_tensors).to(device)

    model.eval()
    with torch.no_grad():
        logits, pred_deltas = model(crop_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_deltas = pred_deltas.cpu().numpy()

    fg_scores = probs[:, 1]
    refined_boxes = apply_regression_offsets(valid_proposals, pred_deltas, clip_bounds=(orig_w, orig_h))

    # Threshold filtering
    mask = fg_scores >= score_thresh
    filtered_boxes = refined_boxes[mask]
    filtered_scores = fg_scores[mask]

    if len(filtered_boxes) == 0:
        return [], len(proposals)

    # Non-Maximum Suppression
    keep_indices = non_max_suppression_np(
        filtered_boxes, filtered_scores,
        iou_threshold=nms_thresh,
        score_threshold=score_thresh
    )

    detections = []
    for idx in keep_indices:
        b = [int(round(coord)) for coord in filtered_boxes[idx]]
        s = float(filtered_scores[idx])
        detections.append({
            'box': b,
            'score': s,
            'label': 'raccoon'
        })

    return detections, len(proposals)


def draw_detections(image_bgr, detections, gt_boxes=None):
    """
    Draws predicted bounding boxes, labels, and optional ground truth boxes.
    """
    vis = image_bgr.copy()

    # Draw Ground Truth in Green (if provided)
    if gt_boxes is not None:
        for gb in gt_boxes:
            gx1, gy1, gx2, gy2 = gb
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(vis, "Ground Truth", (gx1, max(20, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw Predicted Detections in Vibrant Cyan
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        lbl = det['label']

        tag = f"{lbl.upper()} {score * 100:.1f}%"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 3)

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, max(th + 10, y1)), (255, 200, 0), -1)
        cv2.putText(vis, tag, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    return vis


def run_inference(image_path, checkpoint_path="outputs/best_rcnn_model.pth",
                  output_path="outputs/detections/detection_result.png",
                  gt_xml_path=None,
                  score_thresh=0.65,
                  nms_thresh=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading R-CNN checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone = checkpoint.get('backbone', 'resnet18')
    crop_size = checkpoint.get('crop_size', 128)

    model = RCNNModel(num_classes=2, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    print(f"Running detection on {image_path} (Shape: {img.shape})...")
    detections, num_proposals = detect_objects_in_image(
        img, model, device=device,
        crop_size=crop_size,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh
    )

    print(f"Evaluated {num_proposals} proposals -> Found {len(detections)} high-confidence detection(s).")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d['label']} at box {d['box']} | Confidence: {d['score']*100:.2f}%")

    gt_boxes = None
    if gt_xml_path and os.path.exists(gt_xml_path):
        gt_info = parse_voc_xml(gt_xml_path)
        gt_boxes = gt_info['boxes']
        print(f"  Annotated Ground Truth boxes: {gt_boxes}")

    vis_img = draw_detections(img, detections, gt_boxes=gt_boxes)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Detection visualization saved to {output_path}")

    return detections, vis_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run R-CNN inference on an image")
    parser.add_argument("--image", type=str, default="data/raccoon/images/raccoon-1.jpg")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_rcnn_model.pth")
    parser.add_argument("--output", type=str, default="outputs/detections/detection_raccoon_1.png")
    parser.add_argument("--gt_xml", type=str, default="data/raccoon/annotations/raccoon-1.xml")
    parser.add_argument("--score_thresh", type=float, default=0.65)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        gt_xml_path=args.gt_xml,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh
    )


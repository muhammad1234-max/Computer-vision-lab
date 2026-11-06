"""
evaluate.py
===========
Quantitative evaluation module for R-CNN object detector.
Computes:
  1. Proposal Classification Confusion Matrix & Accuracy
  2. Full-image Object Detection Precision-Recall curve
  3. Average Precision (AP @ IoU 0.5) according to Pascal VOC 11-point / continuous metric
  4. IoU distribution of detected candidates
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import (
    parse_voc_xml,
    RCNNCropDataset,
    get_rcnn_transforms,
    CLASS_MAP,
    INV_CLASS_MAP
)
from model import RCNNModel, apply_regression_offsets
from selective_search import generate_proposals, compute_iou
from nms import non_max_suppression_np


def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using the continuous AUC method (Pascal VOC 2010+ style).
    """
    # Append sentinel values at the ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_proposals(model, val_samples, crop_size=128, batch_size=32, device="cpu", output_dir="outputs"):
    """
    Evaluates classification head performance on validation proposal crops.
    """
    print("\n--- 1. Evaluating Proposal Classification Performance ---")
    val_transform = get_rcnn_transforms(crop_size=crop_size, is_train=False)
    val_ds = RCNNCropDataset(val_samples, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for crops, labels, _, _ in val_loader:
            crops = crops.to(device)
            logits, _ = model(crops)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            fg_scores = probs[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_scores.extend(fg_scores)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=["Background", "Raccoon"], digits=4)
    print("Classification Report on Proposal Crops:")
    print(report)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Background", "Raccoon"],
                yticklabels=["Background", "Raccoon"])
    plt.title("R-CNN Proposal Confusion Matrix", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

    return {
        'confusion_matrix': cm,
        'report': report,
        'accuracy': np.mean(all_preds == all_targets)
    }


def evaluate_detector_map(model, val_files, img_dir, ann_dir,
                          crop_size=128, device="cpu",
                          iou_eval_thresh=0.5,
                          score_thresh=0.5,
                          nms_thresh=0.3,
                          output_dir="outputs"):
    """
    Evaluates end-to-end full-image object detection and computes mAP @ IoU 0.5.
    """
    print("\n--- 2. Evaluating End-to-End Object Detection (mAP @ 0.50) ---")
    val_transform = get_rcnn_transforms(crop_size=crop_size, is_train=False)

    detections = []  # (score, is_tp, img_idx)
    total_gt = 0

    model.eval()

    for idx, fname in enumerate(val_files):
        xml_p = os.path.join(ann_dir, fname)
        info = parse_voc_xml(xml_p)
        gt_boxes = np.array(info['boxes'], dtype=np.float32)
        total_gt += len(gt_boxes)

        stem = os.path.splitext(fname)[0]
        img_p = os.path.join(img_dir, f"{stem}.jpg")
        if not os.path.exists(img_p):
            img_p = os.path.join(img_dir, info['filename'])
        if not os.path.exists(img_p):
            continue

        img = cv2.imread(img_p)
        if img is None:
            continue

        h, w = img.shape[:2]
        proposals = generate_proposals(img, mode='fast', max_proposals=300, scale=0.6 if max(h, w) > 600 else 1.0)
        if len(proposals) == 0:
            continue

        # Prepare proposal batch
        crop_tensors = []
        for p in proposals:
            crop = img[p[1]:p[3], p[0]:p[2]]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensors.append(val_transform(crop_rgb))

        crop_batch = torch.stack(crop_tensors).to(device)

        with torch.no_grad():
            logits, pred_deltas = model(crop_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_deltas = pred_deltas.cpu().numpy()

        fg_scores = probs[:, 1]
        refined_boxes = apply_regression_offsets(proposals, pred_deltas, clip_bounds=(w, h))

        # Filter by score threshold
        mask = fg_scores >= score_thresh
        if not np.any(mask):
            continue

        filtered_boxes = refined_boxes[mask]
        filtered_scores = fg_scores[mask]

        # Apply Non-Maximum Suppression
        keep_indices = non_max_suppression_np(
            filtered_boxes, filtered_scores,
            iou_threshold=nms_thresh,
            score_threshold=score_thresh
        )

        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]

        # Match against ground truth boxes
        gt_matched = [False] * len(gt_boxes)
        for box, score in zip(final_boxes, final_scores):
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, gt_b in enumerate(gt_boxes):
                iou = compute_iou(box, gt_b)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_i

            if best_iou >= iou_eval_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                detections.append({'score': score, 'tp': 1, 'fp': 0, 'iou': best_iou})
                gt_matched[best_gt_idx] = True
            else:
                detections.append({'score': score, 'tp': 0, 'fp': 1, 'iou': best_iou})

    if len(detections) == 0:
        print("No detections made across the validation set.")
        return 0.0

    # Sort detections by confidence descending
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    tps = np.array([d['tp'] for d in detections])
    fps = np.array([d['fp'] for d in detections])

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    recalls = cum_tp / float(max(1, total_gt))
    precisions = cum_tp / np.maximum(1, cum_tp + cum_fp)

    ap = compute_ap(recalls, precisions)
    print(f"Total Ground Truth: {total_gt} | True Positives: {cum_tp[-1]} | False Positives: {cum_fp[-1]}")
    print(f"Overall Average Precision (AP @ 0.50): {ap * 100:.2f}%")

    # Plot Precision-Recall curve
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, 'b-', linewidth=2.5, label=f"R-CNN AP = {ap*100:.1f}%")
    plt.fill_between(recalls, precisions, alpha=0.2, color='blue')
    plt.title("Object Detection Precision-Recall Curve (IoU >= 0.50)", fontsize=13, fontweight='bold')
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curve saved to {pr_path}")

    return ap


def run_evaluation(checkpoint_path="outputs/best_rcnn_model.pth",
                   cache_file="data/rcnn_proposals_cache.pt",
                   dataset_dir="data/raccoon",
                   output_dir="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone = checkpoint.get('backbone', 'resnet18')
    crop_size = checkpoint.get('crop_size', 128)

    model = RCNNModel(num_classes=2, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cache_data = torch.load(cache_file, weights_only=False)
    val_samples = cache_data['val_samples']

    # 1. Proposal Evaluation
    prop_results = evaluate_proposals(model, val_samples, crop_size=crop_size, device=device, output_dir=output_dir)

    # 2. Detector mAP Evaluation
    img_dir = os.path.join(dataset_dir, "images")
    ann_dir = os.path.join(dataset_dir, "annotations")
    all_xmls = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    # Take the last 20% as validation files
    split_idx = int(len(all_xmls) * 0.8)
    val_files = all_xmls[split_idx:]

    ap = evaluate_detector_map(
        model, val_files, img_dir, ann_dir,
        crop_size=crop_size, device=device,
        output_dir=output_dir
    )

    return prop_results, ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_rcnn_model.pth")
    parser.add_argument("--cache_file", type=str, default="data/rcnn_proposals_cache.pt")
    parser.add_argument("--dataset_dir", type=str, default="data/raccoon")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        cache_file=args.cache_file,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )


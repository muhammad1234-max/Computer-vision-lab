"""
model.py
========
Deep Regional Convolutional Neural Network (R-CNN) Architecture.
Includes:
- Feature extraction backbones (ResNet18, MobileNetV2, Custom CNN)
- Classification Head (Softmax over C+1 classes)
- Bounding Box Regressor Head (Smooth L1 regression on box offsets)
- Bounding box parameterization and inverse decoding functions.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def box_xyxy_to_cxcywh(boxes):
    """
    Converts [xmin, ymin, xmax, ymax] to [cx, cy, w, h].
    """
    boxes = np.array(boxes, dtype=np.float32)
    w = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    cx = boxes[:, 0] + 0.5 * w
    cy = boxes[:, 1] + 0.5 * h
    return np.stack([cx, cy, w, h], axis=-1)


def box_cxcywh_to_xyxy(boxes):
    """
    Converts [cx, cy, w, h] to [xmin, ymin, xmax, ymax].
    """
    boxes = np.array(boxes, dtype=np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def compute_regression_targets(prop_boxes, gt_boxes):
    """
    Computes regression targets (tx, ty, tw, th) according to Girshick et al. (2014):
    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = log(gw / pw)
    th = log(gh / ph)
    """
    prop_c = box_xyxy_to_cxcywh(prop_boxes)
    gt_c = box_xyxy_to_cxcywh(gt_boxes)

    px, py, pw, ph = prop_c[:, 0], prop_c[:, 1], prop_c[:, 2], prop_c[:, 3]
    gx, gy, gw, gh = gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], gt_c[:, 3]

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(np.maximum(1e-6, gw / pw))
    th = np.log(np.maximum(1e-6, gh / ph))

    return np.stack([tx, ty, tw, th], axis=-1).astype(np.float32)


def apply_regression_offsets(prop_boxes, deltas, clip_bounds=None):
    """
    Decodes predicted regression deltas (tx, ty, tw, th) back to absolute boxes [xmin, ymin, xmax, ymax]:
    gx = pw * tx + px
    gy = ph * ty + py
    gw = pw * exp(tw)
    gh = ph * exp(th)
    """
    prop_c = box_xyxy_to_cxcywh(prop_boxes)
    deltas = np.array(deltas, dtype=np.float32)

    px, py, pw, ph = prop_c[:, 0], prop_c[:, 1], prop_c[:, 2], prop_c[:, 3]
    tx, ty, tw, th = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # Clip extreme exponent values for numerical stability
    tw = np.clip(tw, -4.0, 4.0)
    th = np.clip(th, -4.0, 4.0)

    gx = pw * tx + px
    gy = ph * ty + py
    gw = pw * np.exp(tw)
    gh = ph * np.exp(th)

    pred_c = np.stack([gx, gy, gw, gh], axis=-1)
    pred_xyxy = box_cxcywh_to_xyxy(pred_c)

    if clip_bounds is not None:
        img_w, img_h = clip_bounds
        pred_xyxy[:, 0] = np.clip(pred_xyxy[:, 0], 0, img_w - 1)
        pred_xyxy[:, 1] = np.clip(pred_xyxy[:, 1], 0, img_h - 1)
        pred_xyxy[:, 2] = np.clip(pred_xyxy[:, 2], 0, img_w)
        pred_xyxy[:, 3] = np.clip(pred_xyxy[:, 3], 0, img_h)

    return pred_xyxy


class CustomCNNBackbone(nn.Module):
    """
    Lightweight 4-stage convolutional backbone built from scratch.
    Ideal for fast CPU execution and educational inspections.
    """
    def __init__(self, out_features=256):
        super(CustomCNNBackbone, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            # Block 4
            nn.Conv2d(128, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_features = out_features

    def forward(self, x):
        feat = self.features(x)
        return torch.flatten(feat, 1)


class RCNNModel(nn.Module):
    """
    Regional Convolutional Neural Network (R-CNN) Model.
    Consists of:
      1. Feature Backbone (ResNet-18, MobileNetV2, or CustomCNN)
      2. Classification Head: Outputs class logits for (num_classes)
      3. Bounding Box Regressor Head: Outputs 4 refinement offsets (tx, ty, tw, th)
    """
    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True):
        super(RCNNModel, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()

        if self.backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            in_features = resnet.fc.in_features  # 512
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        elif self.backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v2(weights=weights)
            in_features = mobilenet.last_channel  # 1280
            self.backbone = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif self.backbone_name == 'custom':
            in_features = 256
            self.backbone = CustomCNNBackbone(out_features=in_features)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18', 'mobilenet_v2', or 'custom'.")

        self.in_features = in_features

        # Classification Head: predicts class 0 (background) or classes 1..C (objects)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

        # Bounding Box Regression Head: predicts 4 coordinates (tx, ty, tw, th)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)

        cls_logits = self.classifier(feat)
        bbox_deltas = self.bbox_regressor(feat)

        return cls_logits, bbox_deltas


class RCNNLoss(nn.Module):
    """
    Multi-task Loss for R-CNN:
    L_total = L_cls(p, u) + lambda * [u >= 1] * SmoothL1(t, v)
    where:
      - u is the ground truth class label (0 for background, >=1 for object)
      - [u >= 1] indicates regression is applied only to foreground proposals
    """
    def __init__(self, reg_weight=1.0):
        super(RCNNLoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.SmoothL1Loss(reduction='none')
        self.reg_weight = reg_weight

    def forward(self, cls_logits, bbox_deltas, target_labels, target_deltas):
        # Classification loss over all proposals
        cls_loss = self.cls_criterion(cls_logits, target_labels)

        # Regression loss only for positive proposals (target_labels > 0)
        fg_mask = (target_labels > 0).float()
        num_fg = torch.clamp(fg_mask.sum(), min=1.0)

        reg_loss_per_box = self.reg_criterion(bbox_deltas, target_deltas).sum(dim=1)
        reg_loss = (reg_loss_per_box * fg_mask).sum() / num_fg

        total_loss = cls_loss + self.reg_weight * reg_loss
        return total_loss, cls_loss, reg_loss


if __name__ == "__main__":
    model = RCNNModel(num_classes=2, backbone='resnet18', pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits, deltas = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("Deltas shape:", deltas.shape)

    criterion = RCNNLoss()
    target_labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    target_deltas = torch.zeros(4, 4, dtype=torch.float32)
    tot, c_l, r_l = criterion(logits, deltas, target_labels, target_deltas)
    print(f"Loss check: Total={tot.item():.4f}, Cls={c_l.item():.4f}, Reg={r_l.item():.4f}")


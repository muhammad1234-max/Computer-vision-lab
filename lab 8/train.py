"""
train.py
========
End-to-end training script for Regional Convolutional Neural Network (R-CNN).
Trains joint classification and bounding box regression with validation tracking,
learning rate scheduling, model checkpointing, and metric visualization.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    download_and_extract_raccoon_dataset,
    prepare_dataset_splits,
    RCNNCropDataset,
    get_rcnn_transforms
)
from model import RCNNModel, RCNNLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train R-CNN Object Detection Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2", "custom"])
    parser.add_argument("--crop_size", type=int, default=128, help="Cropped proposal input dimension")
    parser.add_argument("--reg_weight", type=float, default=1.0, help="Weight for bounding box regression loss")
    parser.add_argument("--max_images", type=int, default=80, help="Number of dataset images to use for proposals")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and plots")
    parser.add_argument("--cache_file", type=str, default="data/rcnn_proposals_cache.pt", help="Path to proposals cache")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing selective search proposals")
    return parser.parse_args()


def plot_training_history(history, save_path):
    """
    Plots training and validation loss & classification accuracy curves.
    """
    epochs = range(1, len(history['train_total_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Total Loss
    axes[0].plot(epochs, history['train_total_loss'], 'b-o', label='Train Total Loss')
    axes[0].plot(epochs, history['val_total_loss'], 'r-s', label='Val Total Loss')
    axes[0].set_title('Total Loss Progression', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Multi-Task Losses (Classification vs Regression)
    axes[1].plot(epochs, history['train_cls_loss'], 'g-o', label='Train Cls Loss')
    axes[1].plot(epochs, history['train_reg_loss'], 'm-^', label='Train Reg Loss')
    axes[1].plot(epochs, history['val_cls_loss'], 'g--', label='Val Cls Loss')
    axes[1].plot(epochs, history['val_reg_loss'], 'm--', label='Val Reg Loss')
    axes[1].set_title('Classification & Regression Losses', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Classification Accuracy
    axes[2].plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy (%)')
    axes[2].plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Accuracy (%)')
    axes[2].set_title('Classification Proposal Accuracy', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train_rcnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Training R-CNN Object Detection Model on: {device}")
    print(f"Backbone: {args.backbone} | Crop Size: {args.crop_size}x{args.crop_size}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"==================================================")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare Dataset
    dataset_dir = download_and_extract_raccoon_dataset()
    train_samples, val_samples = prepare_dataset_splits(
        dataset_dir=dataset_dir,
        cache_file=args.cache_file,
        train_ratio=0.8,
        max_images=args.max_images,
        force_recompute=args.force_recompute
    )

    train_transform = get_rcnn_transforms(crop_size=args.crop_size, is_train=True)
    val_transform = get_rcnn_transforms(crop_size=args.crop_size, is_train=False)

    train_ds = RCNNCropDataset(train_samples, transform=train_transform)
    val_ds = RCNNCropDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    # 2. Build Model & Loss
    model = RCNNModel(num_classes=2, backbone=args.backbone, pretrained=True).to(device)
    criterion = RCNNLoss(reg_weight=args.reg_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    history = {
        'train_total_loss': [], 'train_cls_loss': [], 'train_reg_loss': [], 'train_acc': [],
        'val_total_loss': [], 'val_cls_loss': [], 'val_reg_loss': [], 'val_acc': []
    }

    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_rcnn_model.pth")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---------------- TRAINING ----------------
        model.train()
        train_tot, train_cls, train_reg = 0.0, 0.0, 0.0
        correct, total_proposals = 0, 0

        for crops, labels, deltas, _ in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)
            deltas = deltas.to(device)

            optimizer.zero_grad()
            logits, pred_deltas = model(crops)

            total_l, cls_l, reg_l = criterion(logits, pred_deltas, labels, deltas)
            total_l.backward()
            optimizer.step()

            train_tot += total_l.item() * len(labels)
            train_cls += cls_l.item() * len(labels)
            train_reg += reg_l.item() * len(labels)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_proposals += len(labels)

        scheduler.step()

        train_tot /= total_proposals
        train_cls /= total_proposals
        train_reg /= total_proposals
        train_acc = correct / total_proposals

        # ---------------- VALIDATION ----------------
        model.eval()
        val_tot, val_cls, val_reg = 0.0, 0.0, 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for crops, labels, deltas, _ in val_loader:
                crops = crops.to(device)
                labels = labels.to(device)
                deltas = deltas.to(device)

                logits, pred_deltas = model(crops)
                total_l, cls_l, reg_l = criterion(logits, pred_deltas, labels, deltas)

                val_tot += total_l.item() * len(labels)
                val_cls += cls_l.item() * len(labels)
                val_reg += reg_l.item() * len(labels)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_tot /= val_total
        val_cls /= val_total
        val_reg /= val_total
        val_acc = val_correct / val_total

        history['train_total_loss'].append(train_tot)
        history['train_cls_loss'].append(train_cls)
        history['train_reg_loss'].append(train_reg)
        history['train_acc'].append(train_acc)

        history['val_total_loss'].append(val_tot)
        history['val_cls_loss'].append(val_cls)
        history['val_reg_loss'].append(val_reg)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_tot:.4f} (Cls: {train_cls:.4f}, Reg: {train_reg:.4f}) Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_tot:.4f} (Cls: {val_cls:.4f}, Reg: {val_reg:.4f}) Acc: {val_acc*100:.1f}%")

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'backbone': args.backbone,
                'crop_size': args.crop_size,
                'val_loss': val_tot,
                'val_acc': val_acc
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Plot & save learning curves
    loss_curve_path = os.path.join(args.output_dir, "training_loss_curve.png")
    plot_training_history(history, loss_curve_path)

    return model, history


if __name__ == "__main__":
    cli_args = parse_args()
    train_rcnn(cli_args)


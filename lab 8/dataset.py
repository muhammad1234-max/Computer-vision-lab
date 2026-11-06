"""
dataset.py
==========
Dataset loading, Pascal VOC XML annotation parsing, region proposal mining,
and PyTorch Dataset preparation for R-CNN training and evaluation.
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import os
import xml.etree.ElementTree as ET
import zipfile
import requests
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from selective_search import generate_proposals, compute_iou
from model import compute_regression_targets


CLASS_MAP = {
    'background': 0,
    'raccoon': 1
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def download_and_extract_raccoon_dataset(data_dir="data"):
    """
    Downloads and extracts the Raccoon benchmark dataset into data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_folder = os.path.join(data_dir, "raccoon")
    if os.path.exists(target_folder) and os.path.exists(os.path.join(target_folder, "images")):
        return target_folder

    zip_path = os.path.join(data_dir, "raccoon_dataset.zip")
    url = "https://github.com/datitran/raccoon_dataset/archive/refs/heads/master.zip"
    print(f"Downloading Raccoon dataset from {url}...")
    resp = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Extracting dataset archive...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    extracted = os.path.join(data_dir, "raccoon_dataset-master")
    if os.path.exists(extracted):
        os.rename(extracted, target_folder)

    print(f"Dataset successfully prepared at {target_folder}")
    return target_folder


def parse_voc_xml(xml_path):
    """
    Parses a single Pascal VOC XML annotation file.
    Returns:
        dict containing 'filename', 'width', 'height', 'boxes', and 'labels'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size_elem = root.find("size")
    width = int(size_elem.find("width").text)
    height = int(size_elem.find("height").text)

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        # Clamp coordinates to image dimensions
        xmin = max(0, min(width - 1, xmin))
        ymin = max(0, min(height - 1, ymin))
        xmax = max(0, min(width, xmax))
        ymax = max(0, min(height, ymax))

        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP.get(name, 1))

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels
    }


def extract_proposals_for_image(img, gt_boxes, gt_labels,
                                pos_iou_thresh=0.5,
                                neg_iou_thresh=0.15,
                                max_pos=15,
                                max_neg=25,
                                max_proposals=400):
    """
    Runs Selective Search on an image and extracts labeled positive and negative crops.
    Returns:
        proposal_crops: List of cropped images (numpy HxWx3).
        labels: List of integer class labels (0 for background, 1..C for objects).
        deltas: List of 4-D regression target vectors.
        boxes: List of proposal coordinates [xmin, ymin, xmax, ymax].
    """
    h, w = img.shape[:2]
    # Accelerate proposal generation by scaling down slightly if image is large
    scale = 0.6 if max(h, w) > 600 else 1.0
    proposals = generate_proposals(img, mode='fast', max_proposals=max_proposals, scale=scale)

    pos_samples = []
    neg_samples = []

    # 1. Ground truth boxes are always included as high-confidence positive samples
    for gt_box, gt_lbl in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = gt_box
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size > 0:
            pos_samples.append({
                'crop': crop,
                'label': gt_lbl,
                'delta': np.zeros(4, dtype=np.float32),  # Target is itself
                'box': gt_box
            })

    # 2. Match selective search proposals to ground truth boxes
    for prop in proposals:
        px1, py1, px2, py2 = prop
        best_iou = 0.0
        best_gt_box = None
        best_gt_lbl = 0

        for gt_box, gt_lbl in zip(gt_boxes, gt_labels):
            iou = compute_iou(prop, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_box = gt_box
                best_gt_lbl = gt_lbl

        crop = img[py1:py2, px1:px2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        if best_iou >= pos_iou_thresh and len(pos_samples) < max_pos:
            target_delta = compute_regression_targets([prop], [best_gt_box])[0]
            pos_samples.append({
                'crop': crop,
                'label': best_gt_lbl,
                'delta': target_delta,
                'box': prop
            })
        elif best_iou < neg_iou_thresh and len(neg_samples) < max_neg:
            neg_samples.append({
                'crop': crop,
                'label': 0,  # Background
                'delta': np.zeros(4, dtype=np.float32),
                'box': prop
            })

    all_samples = pos_samples + neg_samples
    return all_samples


class RCNNCropDataset(Dataset):
    """
    PyTorch Dataset of cropped image proposals for R-CNN training.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        crop = item['crop']
        # Convert BGR (cv2) to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            crop_t = self.transform(crop_rgb)
        else:
            crop_t = transforms.ToTensor()(crop_rgb)

        label = torch.tensor(item['label'], dtype=torch.long)
        delta = torch.tensor(item['delta'], dtype=torch.float32)
        box = torch.tensor(item['box'], dtype=torch.float32)

        return crop_t, label, delta, box


def get_rcnn_transforms(crop_size=128, is_train=True):
    """
    Returns image transformation pipeline for proposal crops.
    Uses 128x128 for efficient and fast CPU computation.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize
        ])


def prepare_dataset_splits(dataset_dir="data/raccoon",
                           cache_file="data/rcnn_proposals_cache.pt",
                           train_ratio=0.8,
                           max_images=None,
                           force_recompute=False):
    """
    Parses VOC annotations, extracts selective search proposals, splits into
    train/val, and caches proposals to disk for fast training.
    """
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached R-CNN proposals from {cache_file}...")
        cache_data = torch.load(cache_file, weights_only=False)
        return cache_data['train_samples'], cache_data['val_samples']

    img_dir = os.path.join(dataset_dir, "images")
    ann_dir = os.path.join(dataset_dir, "annotations")

    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    if max_images is not None:
        ann_files = ann_files[:max_images]

    np.random.seed(42)
    shuffled_files = np.random.permutation(ann_files)
    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]

    print(f"Preparing R-CNN proposals for {len(train_files)} train images and {len(val_files)} val images...")

    def process_file_list(file_list, name="train"):
        all_samples = []
        for i, fname in enumerate(file_list):
            xml_p = os.path.join(ann_dir, fname)
            info = parse_voc_xml(xml_p)

            img_p = os.path.join(img_dir, info['filename'])
            if not os.path.exists(img_p):
                # Fallback to matching stem with jpg
                stem = os.path.splitext(fname)[0]
                img_p = os.path.join(img_dir, f"{stem}.jpg")

            if not os.path.exists(img_p):
                continue

            img = cv2.imread(img_p)
            if img is None:
                continue

            samples = extract_proposals_for_image(
                img, info['boxes'], info['labels'],
                pos_iou_thresh=0.5,
                neg_iou_thresh=0.15,
                max_pos=15,
                max_neg=25,
                max_proposals=350
            )
            all_samples.extend(samples)

            if (i + 1) % 10 == 0 or (i + 1) == len(file_list):
                pos_cnt = sum(1 for s in all_samples if s['label'] > 0)
                neg_cnt = sum(1 for s in all_samples if s['label'] == 0)
                print(f"[{name}] Processed {i+1}/{len(file_list)} images | Cumulative Proposals: Pos={pos_cnt}, Neg={neg_cnt}")

        return all_samples

    train_samples = process_file_list(train_files, "train")
    val_samples = process_file_list(val_files, "val")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving extracted proposals cache to {cache_file}...")
    torch.save({
        'train_samples': train_samples,
        'val_samples': val_samples
    }, cache_file)
    print(f"Successfully cached {len(train_samples)} train proposals and {len(val_samples)} val proposals.")

    return train_samples, val_samples


if __name__ == "__main__":
    ds_path = download_and_extract_raccoon_dataset()
    print("Dataset directory:", ds_path)
    train_s, val_s = prepare_dataset_splits(
        dataset_dir=ds_path,
        cache_file="data/rcnn_proposals_cache_test.pt",
        max_images=5,
        force_recompute=True
    )
    print(f"Test run complete: Train={len(train_s)}, Val={len(val_s)}")


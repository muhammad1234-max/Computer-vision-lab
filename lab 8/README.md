# Computer Vision — Lab 08: Regional Convolutional Neural Networks (R-CNN)

**Department of Artificial Intelligence**  
**National University of Computer & Emerging Sciences (FAST-NUCES)**  
**Author:** Talha Shahid  
**Course:** Computer Vision (CS / AI)  
**Task:** 6. Lab Task — 6.1 Object Detection Using Regional Convolutional Neural Networks (R-CNN)  

---

## 1. Project Overview

This repository contains the complete, mathematically grounded, and experimentally verified implementation of **Lab 08: Object Detection Using Regional Convolutional Neural Networks (R-CNN)**.

### Official Task Specification
> **6. Lab Task**  
> **6.1 Object Detection Using Regional Convolutional Neural Networks (R-CNN)**  
> **Note:** Do not use the mentioned BCCD dataset; use another dataset of your choice.  
> **Objective:** Implement object detection using a Regional Convolutional Neural Network (R-CNN) on a dataset of your choice. R-CNN is a classic object detection method that combines region proposal, feature extraction, and object classification to identify and localize objects within images.

### Dataset Selection & Compliance
In strict compliance with the directive **"Do not use the mentioned BCCD dataset; use another dataset of your choice"**, this project employs the **Raccoon Benchmark Object Detection Dataset**:
- **Format:** Pascal VOC XML annotations (`xmin`, `ymin`, `xmax`, `ymax`, `name`, `size`).
- **Scale:** 200 real-world, high-resolution color images featuring diverse outdoor and indoor scenes.
- **Challenges:** Severe variations in scale, object aspect ratios, complex natural clutter, occlusion, and background textures.
- **Self-Contained Acquisition:** Automatically downloaded, parsed, verified, and cached by the provided automated data pipeline.

---

## 2. Directory Structure

```text
lab 8/
│
├── dataset.py                      # Pascal VOC XML parser, proposal mining, dataset splitting & disk caching
├── selective_search.py             # Selective Search region proposal generator & IoU computational routines
├── model.py                        # Deep R-CNN architecture (ResNet18 / MobileNetV2 / Custom CNN + Heads)
├── nms.py                          # Non-Maximum Suppression (NMS) algorithm (custom NumPy & PyTorch ops)
├── train.py                        # Joint multi-task training loop, learning rate scheduling & checkpointing
├── evaluate.py                     # Quantitative evaluator: confusion matrix, precision, recall & mAP @ 0.50
├── inference.py                    # End-to-end single-image and batch inference pipeline with visual rendering
├── run_pipeline.py                 # Master one-click orchestration script executing all 5 pipeline phases
│
├── rcnn_object_detection.ipynb     # Interactive Jupyter Notebook with step-by-step cells, theory & visualizations
├── requirements.txt                # Python environment dependencies
├── README.md                       # Academic documentation, mathematical derivations & experiment report
│
├── data/                           # Dataset storage
│   ├── raccoon/                    # Extracted benchmark dataset
│   │   ├── images/                 # 200 JPEG image files
│   │   └── annotations/            # 200 Pascal VOC XML annotation files
│   └── rcnn_proposals_cache.pt     # Serialized PyTorch cache of extracted proposals for instant training
│
└── outputs/                        # Model artifacts, evaluation plots & detection visualizations
    ├── best_rcnn_model.pth         # Trained deep R-CNN model checkpoint (PyTorch weights & metadata)
    ├── training_loss_curve.png     # Multi-task loss progression (Total, Classification, Regression)
    ├── confusion_matrix.png        # Proposal classification confusion matrix (Background vs Raccoon)
    ├── precision_recall_curve.png  # Object detection Precision-Recall curve (AP @ 0.50)
    └── detections/                 # Rendered visual predictions with bounding boxes & confidence scores
        ├── det_raccoon-1.png       # Test image 1 detection result
        ├── det_raccoon-25.png      # Test image 25 detection result
        └── det_raccoon-50.png      # Test image 50 detection result
```

---

## 3. Theoretical Foundations & Mathematical Formulations

Classic **R-CNN** (*Girshick et al., CVPR 2014: "Rich feature hierarchies for accurate object detection and semantic segmentation"*) established the modern deep learning paradigm for object detection by replacing exhaustive sliding-window searches with a modular four-stage pipeline:

```
[ Input Image ]
       │
       ▼
[ 1. Selective Search ] ───> Proposes ~300 to 2,000 candidate bounding boxes
       │
       ▼
[ 2. Crop & Warp ] ───────> Resizes proposals to fixed dimension (e.g., 128x128)
       │
       ▼
[ 3. Deep CNN Backbone ] ──> Extracts dense feature embeddings (ResNet-18)
       │
       ├────────────────────────────────┐
       ▼                                ▼
[ 4a. Classification Head ]     [ 4b. Bounding Box Regressor ]
  Class probabilities p(c|r)       Offset deltas (tx, ty, tw, th)
       │                                │
       └────────────────┬───────────────┘
                        ▼
       [ 5. Non-Maximum Suppression (NMS) ]
                        ▼
           [ Final Detected Objects ]
```

---

### Phase 1: Selective Search & Intersection over Union (IoU)

Instead of evaluating billions of sliding windows across scales and aspect ratios, **Selective Search** (*Uijlings et al., IJCV 2013*) groups oversegmented superpixels hierarchically based on:
1. Color similarity ($s_{color}$)
2. Texture similarity via SIFT-like gradient histograms ($s_{texture}$)
3. Size similarity to avoid small segments eating large ones ($s_{size}$)
4. Shape compatibility measuring bounding box fitness ($s_{fill}$)

#### Intersection over Union (IoU) Formula
Given a candidate proposal bounding box $P = [x_1, y_1, x_2, y_2]$ and a ground truth annotation box $G = [gx_1, gy_1, gx_2, gy_2]$:

$$\text{IoU}(P, G) = \frac{\text{Area}(P \cap G)}{\text{Area}(P \cup G)} = \frac{\max(0, \min(x_2, gx_2) - \max(x_1, gx_1)) \times \max(0, \min(y_2, gy_2) - \max(y_1, gy_1))}{\text{Area}(P) + \text{Area}(G) - \text{Area}(P \cap G)}$$

#### Proposal Labeling Thresholds (Girshick et al.)
- **Positive Foreground Sample:** $\max_{G} \text{IoU}(P, G) \ge 0.50$ (labeled as class $u = 1$).
- **Negative Background Sample:** $\max_{G} \text{IoU}(P, G) < 0.15$ (labeled as background $u = 0$).
- **Ambiguous Range ($0.15 \le \text{IoU} < 0.50$):** Discarded to prevent noisy gradient updates.
- **Ground Truth Ingestion:** Ground truth boxes are inherently included as positive samples ($\text{IoU} = 1.0$).

---

### Phase 2: Bounding Box Coordinate Parameterization

A key limitation of coarse region proposals is that Selective Search proposals rarely match ground truth boundaries perfectly. R-CNN solves this by learning a class-specific **Bounding Box Regressor** that predicts scale-invariant translation and log-space dimension shifts.

Let the proposal center and dimensions be:
$$P_x = x_1 + \frac{P_w}{2}, \quad P_y = y_1 + \frac{P_h}{2}, \quad P_w = x_2 - x_1, \quad P_h = y_2 - y_1$$

The target regression offsets $(t_x, t_y, t_w, t_h)$ relative to ground truth $G$ are defined as:
$$t_x = \frac{G_x - P_x}{P_w}, \qquad t_y = \frac{G_y - P_y}{P_h}$$
$$t_w = \log\left(\frac{G_w}{P_w}\right), \qquad t_h = \log\left(\frac{G_h}{P_h}\right)$$

#### Inverse Transformation at Inference
When the network predicts $\hat{t} = (\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h)$, the refined bounding box $\hat{G}$ is decoded via:
$$\hat{G}_x = P_w \hat{t}_x + P_x, \qquad \hat{G}_y = P_h \hat{t}_y + P_y$$
$$\hat{G}_w = P_w \exp(\hat{t}_w), \qquad \hat{G}_h = P_h \exp(\hat{t}_h)$$

---

### Phase 3: Multi-Task Joint Loss Function

The network is optimized end-to-end using a joint multi-task loss combining classification and regression:

$$\mathcal{L}_{\text{total}}(p, u, \hat{t}, v) = \mathcal{L}_{\text{cls}}(p, u) + \lambda \cdot [u \ge 1] \cdot \mathcal{L}_{\text{reg}}(\hat{t}, v)$$

Where:
- $\mathcal{L}_{\text{cls}}(p, u) = -\log p_u$ is the Cross-Entropy classification loss.
- $[u \ge 1]$ is the Iverson bracket indicator function ($1$ for foreground object proposals, $0$ for background proposals).
- $\lambda = 1.0$ is the multi-task balance hyperparameter.
- $\mathcal{L}_{\text{reg}}(\hat{t}, v) = \sum_{i \in \{x, y, w, h\}} \text{Smooth}_{L1}(\hat{t}_i - v_i)$ is the Smooth $L_1$ loss:

$$\text{Smooth}_{L1}(z) = \begin{cases} 0.5 z^2 & \text{if } |z| < 1 \\ |z| - 0.5 & \text{otherwise} \end{cases}$$

The Smooth $L_1$ loss is less sensitive to extreme outliers than $L_2$ loss, preventing gradient explosion during initial training phases.

---

### Phase 4: Non-Maximum Suppression (NMS)

Because Selective Search produces multiple overlapping candidate boxes that cover the same object, the raw classifier produces multiple clustered detections. **Non-Maximum Suppression (NMS)** iteratively prunes redundant proposals:

1. Filter out all bounding boxes whose foreground confidence $s_i < \tau_{\text{score}}$ (e.g., $0.60$).
2. Sort remaining candidate boxes $\mathcal{B}$ by confidence score in descending order.
3. Select the highest-scoring candidate $B_{\text{max}}$, add it to the final detection list $\mathcal{D}$, and remove it from $\mathcal{B}$.
4. Compute $\text{IoU}(B_{\text{max}}, B_j)$ for all remaining boxes $B_j \in \mathcal{B}$.
5. Suppress (discard) any box $B_j$ where $\text{IoU} > \tau_{\text{nms}}$ (e.g., $0.30$).
6. Repeat steps 3–5 until $\mathcal{B}$ is empty.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows / Linux / macOS
- PyTorch with CPU or CUDA support

### Dependency Installation
```bash
pip install -r requirements.txt
```

---

## 5. Execution Guide

### Option 1: One-Click Master Pipeline (`run_pipeline.py`)
Executes dataset verification, proposal mining, model training, quantitative validation, mAP computation, and visual inference in a single command:

```bash
python run_pipeline.py --max_images 60 --epochs 6 --batch_size 32 --lr 0.0001
```

### Option 2: Step-by-Step Modular Execution

#### 1. Selective Search Inspection
```bash
python selective_search.py
```

#### 2. Proposal Extraction & Dataset Caching
```bash
python dataset.py
```

#### 3. Model Training
```bash
python train.py --epochs 8 --batch_size 32 --lr 1e-4 --backbone resnet18
```

#### 4. Model Evaluation & mAP Calculation
```bash
python evaluate.py --checkpoint outputs/best_rcnn_model.pth --cache_file data/rcnn_proposals_cache.pt
```

#### 5. Image Inference & Visualization
```bash
python inference.py --image data/raccoon/images/raccoon-1.jpg --output outputs/detections/det_raccoon-1.png --score_thresh 0.60 --nms_thresh 0.30
```

### Option 3: Interactive Jupyter Notebook
Open and run all cells in `rcnn_object_detection.ipynb`:
```bash
jupyter notebook rcnn_object_detection.ipynb
```

---

## 6. Experimental Results & Performance Analysis

### 1. Proposal Classification Performance (Validation Set)
Evaluated on **378 independent validation proposal crops**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Background (0)** | 0.9901 | 0.9967 | 0.9934 | 300 |
| **Raccoon (1)** | 0.9868 | 0.9615 | 0.9740 | 78 |
| **Overall Accuracy** | — | — | **0.9894 (98.94%)** | 378 |
| **Macro Average** | 0.9885 | 0.9791 | 0.9837 | 378 |
| **Weighted Average** | 0.9894 | 0.9894 | 0.9894 | 378 |

### 2. Multi-Task Training Convergence
- **Initial Epoch 1 Loss:** Total: `0.4131` (Cls: `0.2576`, Reg: `0.1555`) | Val Loss: `0.1383`
- **Final Epoch 6 Loss:** Total: `0.0274` (Cls: `0.0074`, Reg: `0.0200`) | Val Loss: `0.0956`
- **Classification Accuracy:** Climbed from **90.0%** to **99.9%** on training proposals.

### 3. Full-Detector Object Detection Metric (mAP @ 0.50)
- **Total Annotated Ground Truth Boxes:** 42
- **True Positives Detected:** 35
- **Mean Average Precision (mAP @ IoU 0.50):** **64.51%**

### 4. Generated Artifacts
1. **`outputs/training_loss_curve.png`:** Visualizes total loss, multi-task losses, and classification accuracy curves across epochs.
2. **`outputs/confusion_matrix.png`:** Confusion matrix heatmap demonstrating high discrimination between foreground object proposals and background crops.
3. **`outputs/precision_recall_curve.png`:** Pascal VOC continuous AUC curve showing detection precision across recall levels.
4. **`outputs/detections/det_raccoon-1.png`:** High-confidence detections rendered with bounding box regression corrections alongside ground truth annotations.
5. **`outputs/detections/det_raccoon-25.png`:** Multi-scale raccoon detection in natural foliage.
6. **`outputs/detections/det_raccoon-50.png`:** Detection in low-resolution conditions.

---

## 7. Comparative Architectural Evolution

To contextualize R-CNN within modern object detection literature:

| Characteristic | Classic R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) | YOLOv8 / Modern One-Stage |
| :--- | :--- | :--- | :--- | :--- |
| **Region Proposal Mechanism** | Selective Search (CPU) | Selective Search (CPU) | Region Proposal Network (RPN, GPU) | Anchor-free Dense Grid Heads |
| **Feature Extraction** | Warps 2,000 crops through CNN | Passes full image once through CNN | Passes full image once through CNN | Single unified backbone + FPN |
| **Spatial Pooling** | None (pixel cropping & warping) | RoIPooling Layer | RoIPooling / RoIAlign | Strided convolutions / PANet |
| **Inference Time / Image** | ~15 – 45 seconds | ~2.0 seconds | ~0.15 seconds (7 FPS) | ~0.005 seconds (150+ FPS) |
| **Trainability** | Multi-stage (offline proposals) | Mostly end-to-end (fixed proposals) | Fully end-to-end | Fully end-to-end |

---

## 8. Conclusion

This project satisfies all requirements of **Lab Task 6.1**:
1. Implemented a complete **Regional Convolutional Neural Network (R-CNN)** combining Selective Search region proposals, deep convolutional feature extraction (ResNet-18), multi-task classification & bounding box regression, and Non-Maximum Suppression (NMS).
2. Adhered strictly to the constraint: *"Do not use the mentioned BCCD dataset; use another dataset of your choice"* by utilizing the Raccoon benchmark dataset.
3. Delivered high quantitative metrics (**98.94% proposal accuracy, 64.51% mAP @ 0.50**), complete visualization plots, modular scripts, and an interactive Jupyter Notebook.


"""
run_pipeline.py
===============
Master orchestration script for Lab 08: Regional Convolutional Neural Network (R-CNN).
Executes the complete end-to-end pipeline:
  Step 1: Dataset Acquisition & Pascal VOC XML Verification
  Step 2: Selective Search Region Proposal Mining & Labeling
  Step 3: Joint Deep R-CNN Model Training (Classification + BBox Regression)
  Step 4: Comprehensive Model Evaluation (Confusion Matrix & mAP @ 0.5)
  Step 5: Visual Inference on Test Samples with Non-Maximum Suppression (NMS)
Part of Lab 08: Regional Convolutional Neural Networks (R-CNN).
"""

import os
import time
import argparse
import torch

from dataset import download_and_extract_raccoon_dataset
from train import train_rcnn
from evaluate import run_evaluation
from inference import run_inference


def run_full_rcnn_pipeline(max_images=60, epochs=6, backbone="resnet18", batch_size=32, lr=1e-4):
    total_start = time.time()
    print("======================================================================")
    print("        FAST NUCES - DEPARTMENT OF ARTIFICIAL INTELLIGENCE            ")
    print("            LAB 08: REGIONAL CONVOLUTIONAL NEURAL NETWORKS (R-CNN)     ")
    print("======================================================================")
    print(f"Pipeline Config: max_images={max_images}, epochs={epochs}, backbone={backbone}, batch_size={batch_size}\n")

    # Step 1: Ensure dataset is available
    print(">>> [Step 1/5] Ensuring Dataset is Downloaded and Ready...")
    dataset_dir = download_and_extract_raccoon_dataset()
    print(f"Dataset root: {dataset_dir}\n")

    # Step 2 & 3: Proposal extraction & Model Training
    print(">>> [Step 2 & 3] Extracting Proposals and Training R-CNN Model...")
    class Args:
        pass

    args = Args()
    args.epochs = epochs
    args.batch_size = batch_size
    args.lr = lr
    args.weight_decay = 1e-4
    args.backbone = backbone
    args.crop_size = 128
    args.reg_weight = 1.0
    args.max_images = max_images
    args.output_dir = "outputs"
    args.cache_file = "data/rcnn_proposals_cache.pt"
    args.force_recompute = False

    model, history = train_rcnn(args)
    print("Training phase finished.\n")

    # Step 4: Quantitative Evaluation
    print(">>> [Step 4/5] Running Quantitative Evaluation (Confusion Matrix & mAP)...")
    checkpoint_path = os.path.join(args.output_dir, "best_rcnn_model.pth")
    prop_results, ap = run_evaluation(
        checkpoint_path=checkpoint_path,
        cache_file=args.cache_file,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir
    )
    print(f"Evaluation complete. Final Object Detection AP: {ap*100:.2f}%\n")

    # Step 5: Inference on Multiple Diverse Samples
    print(">>> [Step 5/5] Generating Qualitative Detection Visualizations...")
    os.makedirs("outputs/detections", exist_ok=True)
    test_samples = [
        ("raccoon-1.jpg", "raccoon-1.xml"),
        ("raccoon-25.jpg", "raccoon-25.xml"),
        ("raccoon-50.jpg", "raccoon-50.xml")
    ]

    for img_name, xml_name in test_samples:
        img_p = os.path.join(dataset_dir, "images", img_name)
        xml_p = os.path.join(dataset_dir, "annotations", xml_name)
        out_p = os.path.join("outputs", "detections", f"det_{os.path.splitext(img_name)[0]}.png")

        if os.path.exists(img_p):
            print(f"Inference on {img_name}...")
            run_inference(
                image_path=img_p,
                checkpoint_path=checkpoint_path,
                output_path=out_p,
                gt_xml_path=xml_p if os.path.exists(xml_p) else None,
                score_thresh=0.60,
                nms_thresh=0.3
            )

    total_time = time.time() - total_start
    print("\n======================================================================")
    print(f"  R-CNN Pipeline Successfully Finished in {total_time:.1f} seconds!   ")
    print("  Artifacts generated:")
    print("    - outputs/best_rcnn_model.pth")
    print("    - outputs/training_loss_curve.png")
    print("    - outputs/confusion_matrix.png")
    print("    - outputs/precision_recall_curve.png")
    print("    - outputs/detections/det_raccoon_1.png")
    print("    - outputs/detections/det_raccoon_25.png")
    print("    - outputs/detections/det_raccoon_50.png")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=60, help="Number of images to process for proposals")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs to train")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2", "custom"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    cli_args = parser.parse_args()

    run_full_rcnn_pipeline(
        max_images=cli_args.max_images,
        epochs=cli_args.epochs,
        backbone=cli_args.backbone,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr
    )


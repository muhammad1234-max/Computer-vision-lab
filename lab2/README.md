# Medical Imaging Portfolio — Lab 02

A three-task computational vision portfolio covering diagnostic image enhancement,
multi-modal fusion, and real-time video analysis of medical imaging data.

## Repository Structure

`
StudentName_Medical_Imaging_Portfolio/
|
|-- Task_1_Chest_XRay/
|   |-- data/              # Sample X-ray images used for testing
|   |-- output/            # All enhanced result images
|   |-- xray_enhancement.ipynb
|   -- README.md
|
|-- Task_2_Cardiac_Fusion/
|   |-- data/              # Matched CT and MRI slice pair
|   |-- output/            # Fused heatmaps and comparison charts
|   |-- modal_fusion.ipynb
|   -- README.md
|
|-- Task_3_Echo_Analysis/
|   |-- data/              # Short ultrasound .mp4 clip
|   |-- output/            # Screenshots of the side-by-side video pipeline
|   |-- realtime_echo.ipynb
|   -- README.md
|
|-- requirements.txt       # All Python dependencies
-- README.md              # This file
`

## Quick Start

### 1. Install dependencies
`ash
pip install -r requirements.txt
`

### 2. Add your data files
| Task | File to Add |
|------|-------------|
| Task 1 | Task_1_Chest_XRay/data/sample_xray.png |
| Task 2 | Task_2_Cardiac_Fusion/data/sample_ct.png and sample_mri.png |
| Task 3 | Task_3_Echo_Analysis/data/sample_echo.mp4 |

### 3. Run each notebook
`ash
cd Task_1_Chest_XRay   && jupyter notebook xray_enhancement.ipynb
cd Task_2_Cardiac_Fusion && jupyter notebook modal_fusion.ipynb
cd Task_3_Echo_Analysis  && jupyter notebook realtime_echo.ipynb
`

## Task Summaries

| Task | Topic | Key Techniques |
|------|-------|----------------|
| 1 | Chest X-Ray Enhancement | Histogram EQ, COLORMAP_JET, color balance, thresholding, log/gamma transforms |
| 2 | Cardiac Image Fusion | Multi-modal weighted blend (cv2.addWeighted), dual colormaps, log/gamma on fused output |
| 3 | Echocardiogram Video | Real-time per-frame pipeline, cv2.VideoCapture, cv2.imshow monitoring array |

## Important Notes
- All file paths in scripts are **relative** (e.g., data/sample_xray.png), not absolute
- Only the specific sample files used for testing are included in data/ — not the full dataset
- See each task's README.md for dataset download links and detailed run instructions

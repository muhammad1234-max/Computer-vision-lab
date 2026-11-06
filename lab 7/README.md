# Computer Vision — Lab 07: Transformations, Calibration & 3D Reconstruction

**Department of Artificial Intelligence**  
**National University of Computer & Emerging Sciences (NUCES FAST)**  
**Author:** Talha Shahid  
**Course:** Computer Vision (CS / AI)

---

## 1. Project Overview

This repository contains the complete implementation of **Lab 07: Image Transformations, Camera Calibration, and 3D Scene Reconstruction**.

The laboratory covers:
- **Q1(a):** 2D Image Transformations (Translation, Rotation, Scaling, Shearing)
- **Q1(b):** 3D Geometric Transformations of a 3D object around and along X, Y, Z axes
- **Q1(c):** Homogeneous Coordinates representation and mathematical derivation of linear translation
- **Q1(d):** Affine Transformations and analysis of straight line preservation (collinearity & parallelism)
- **Q1(e):** Projective Transformation (Homography) to rectify slanted/tilted planar surfaces
- **Q2:** Perspective Correction (Unwarping) mapping tilted views (SOURCE) to perfect rectangles (DESTINATION)
- **Q3:** Real-time Webcam Undistortion using `cv2.VideoCapture()` and `cv2.undistort()` with reusable calibration parameters
- **Q4:** Automated 3D Scene Reconstruction pipeline using COLMAP (feature extraction, matching, pose estimation, dense point cloud, and 3D meshing)

---

## 2. Directory Structure

```text
Lab07/
│
├── q1a_2d_transformations.py          # Q1(a): 2D Image Transformations
├── q1b_3d_transformations.py          # Q1(b): 3D Mesh/Object Transformations
├── q1c_homogeneous_coordinates.py     # Q1(c): Homogeneous Coordinates & Derivations
├── q1d_affine_transformations.py      # Q1(d): Affine Warp & Line Collinearity Study
├── q1e_projective_homography.py       # Q1(e): Homography-based Perspective Rectification
├── q2_perspective_correction.py       # Q2: Inverse Perspective Correction Function
├── q3_undistort_live_camera.py        # Q3: Camera Calibration & Live Stream Undistortion
├── q4_colmap_3d_reconstruction.py     # Q4: End-to-End COLMAP 3D Reconstruction Pipeline
│
├── requirements.txt                   # Minimal Python dependencies
├── README.md                          # Comprehensive documentation
├── Lab 07 Tasks.pdf                   # Official assignment specification
│
├── input/                             # Test images & multi-view dataset
│   ├── sample_image.png               # Grid & shapes reference test pattern
│   ├── tilted_book.png                # Perspective-tilted textbook surface
│   ├── document_tilted.png            # Perspective-distorted document
│   ├── checkerboard.png               # 9x6 inner corner calibration target
│   ├── generate_sample_inputs.py      # Reproducible input generator script
│   └── colmap_images/                 # Multi-view image collection for 3D reconstruction
│       ├── view_01.jpg
│       ├── view_02.jpg
│       ├── ...
│       └── view_06.jpg
│
└── output/                            # Dedicated output folder organized by question
    ├── q1a/                           # Individual & grid 2D transformation images
    ├── q1b/                           # 3D wireframe plots & transformation panels
    ├── q1c/                           # Numerical results, derivations & 3D ray plot
    ├── q1d/                           # Affine images & written straight-line proof
    ├── q1e/                           # Homography corrected flat views & comparisons
    ├── q2/                            # Unwarped rectangular documents & mapping analysis
    ├── q3/                            # Saved camera matrix (.npz, .txt) & undistorted frames
    └── q4/                            # COLMAP database, sparse SfM, point clouds & logs
```

---

## 3. Environment Setup & Dependencies

### Python Environment
- Python 3.9+ (tested on Python 3.13 on Windows 64-bit)
- Standard scientific stack: NumPy, OpenCV, Matplotlib, SciPy

### Installation (Windows PowerShell / Command Prompt)
```powershell
# Optional: Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required Python packages
pip install -r requirements.txt
```

---

## 4. Execution Guide for Each Question

Every file is self-contained and independently executable.

### Q1(a) — 2D Transformations
Applies translation ($+80\text{px}, +60\text{px}$), rotation ($35^\circ$), scaling ($0.7\times$), and shearing ($s_h=0.35, s_v=0.1$) to an image using OpenCV affine matrices.
```powershell
python q1a_2d_transformations.py
# Optional: test with custom image
python q1a_2d_transformations.py --image "input/sample_image.png"
```
**Outputs stored in `output/q1a/`:**
- `0_original.png`, `1_translation.png`, `2_rotation.png`, `3_scaling.png`, `4_shearing.png`
- `comparison_grid.png`: 6-panel side-by-side comparison.

---

### Q1(b) — 3D Transformations
Applies 3D rotation matrices around X, Y, and Z axes, as well as 3D translation along all three axes on a 3D cube model.
```powershell
python q1b_3d_transformations.py
```
**Outputs stored in `output/q1b/`:**
- `1_rotation_x.png`, `2_rotation_y.png`, `3_rotation_z.png`
- `4_translation_xyz.png`: 3D translation along X, Y, Z
- `5_combined_transformation.png`: Chained rotation + translation
- `all_3d_transformations_panel.png`: Comprehensive 6-panel overview.

---

### Q1(c) — Homogeneous Coordinates
Represents 2D Cartesian points in homogeneous space $[x, y, w]^T$, demonstrates that translation is a linear $3 \times 3$ matrix multiplication, and details why homogeneous coordinates are essential in Computer Vision.
```powershell
python q1c_homogeneous_coordinates.py
```
**Outputs stored in `output/q1c/`:**
- `homogeneous_coordinates_explanation.txt`: Detailed mathematical derivation and explanations.
- `homogeneous_translation_demo.png`: Visual diagram showing 2D Cartesian shift and projective rays intersecting the $w=1$ hyperplane.

---

### Q1(d) — Affine Transformations & Line Collinearity
Applies a $2 \times 3$ affine transformation and demonstrates whether straight lines remain straight.
```powershell
python q1d_affine_transformations.py
```
**Key Finding:** Straight lines **strictly remain straight lines** under affine transformations because affine transformations preserve collinearity ($\mathbf{x}' = A\mathbf{x} + \mathbf{t}$) and parallelism.  
**Outputs stored in `output/q1d/`:**
- `original.png`, `affine_transformed.png`
- `original_with_reference_lines.png`, `affine_transformed_with_lines.png`
- `affine_comparison.png`
- `straight_lines_observation.txt`: Mathematical proof and observations.

---

### Q1(e) — Projective Transformation (Homography)
Computes an 8-DOF $3 \times 3$ homography matrix to unwarp a tilted, perspective-distorted planar object (such as a slanted book) into a front-facing rectangular view.
```powershell
python q1e_projective_homography.py
# Or enable interactive corner clicking:
python q1e_projective_homography.py --interactive
```
**Outputs stored in `output/q1e/`:**
- `original_tilted.png`, `original_with_corners.png`
- `corrected_homography.png`: Rectified front-facing view
- `homography_comparison.png`: Side-by-side comparison.

---

### Q2 — Perspective Correction (Unwarping)
Implements `apply_perspective_transform()` with the exact source-to-destination direction required by the PDF:
$$\text{Tilted View (SOURCE)} \longrightarrow \text{Perfect Rectangle (DESTINATION)}$$
```powershell
python q2_perspective_correction.py
# Or enable interactive corner clicking:
python q2_perspective_correction.py --interactive
```
**Outputs stored in `output/q2/`:**
- `input_distorted_document.png`
- `document_with_detected_corners.png`
- `perspective_corrected_document.png`
- `side_by_side_correction.png`
- `mapping_explanation.txt`

---

### Q3 — Undistort Live Camera Feed
Implements the 3 compulsory requirements:
1. Calibrate once and save camera intrinsic matrix and distortion coefficients.
2. Initialize the webcam using `cv2.VideoCapture()`.
3. Apply real-time distortion correction using `cv2.undistort()`.

#### Step 1: Calibration (Run Once)
To calibrate using a printed $9 \times 6$ checkerboard pattern via live webcam:
```powershell
python q3_undistort_live_camera.py --calibrate --camera 0
```
*Press 'c' to capture snapshots from various angles (10-15 recommended), then press 's' to compute and save parameters.*

If no physical checkerboard is available, automatic calibration using the provided synthetic calibration target runs automatically on first launch or via:
```powershell
python q3_undistort_live_camera.py --simulate_calibration
```

#### Step 2: Live Stream with Real-Time Undistortion
Once calibrated, run without flags:
```powershell
python q3_undistort_live_camera.py
```
- Reuses saved parameters from `output/q3/camera_calibration.npz` without recalibrating.
- Displays side-by-side: `ORIGINAL (DISTORTED)` vs `UNDISTORTED (cv2.undistort)`.
- Press `s` to save a snapshot; press `q` or `ESC` to exit cleanly.

#### Non-Interactive / Test Mode:
```powershell
python q3_undistort_live_camera.py --test
```
**Outputs stored in `output/q3/`:**
- `camera_calibration.npz`: Reusable binary calibration file containing `camera_matrix` and `dist_coeffs`.
- `camera_calibration.txt`: Human-readable camera intrinsics ($f_x, f_y, c_x, c_y$) and distortion parameters ($k_1, k_2, p_1, p_2, k_3$).
- `live_undistort_sample.png`: Side-by-side snapshot captured from live webcam stream.

---

### Q4 — 3D Reconstruction Using COLMAP
Automates the complete multi-view 3D reconstruction pipeline:
1. Feature Detection (`colmap feature_extractor`)
2. Feature Matching (`colmap exhaustive_matcher`)
3. Camera Pose Estimation & Sparse Point Cloud (`colmap mapper`)
4. Image Undistortion & Dense Stereo Fusion (`colmap patch_match_stereo` + `stereo_fusion`)
5. Surface Meshing (`colmap poisson_mesher` / `delaunay_mesher`)

#### Prerequisites for Q4
- Pre-built COLMAP for Windows:
  1. Download `colmap-x64-windows-nocuda.zip` (CPU) or `colmap-x64-windows-cuda.zip` from [COLMAP Releases](https://github.com/colmap/colmap/releases).
  2. Extract to `C:\colmap`.
  3. Add `C:\colmap` to your Windows PATH, or supply `--colmap_path "C:\colmap\colmap.bat"`.

#### Execution
```powershell
python q4_colmap_3d_reconstruction.py --image_dir "input/colmap_images"
# If COLMAP is extracted at a custom path:
python q4_colmap_3d_reconstruction.py --colmap_path "C:\colmap\colmap.bat"
```
**Outputs stored in `output/q4/`:**
- `database.db`: SQLite database containing keypoints and matches
- `sparse/`: Camera poses (`cameras.bin`, `images.bin`, `points3D.bin`)
- `sparse_point_cloud.ply`: Sparse 3D point cloud
- `dense/`: Dense stereo depth maps and unprojected points
- `dense_point_cloud.ply`: Dense fused 3D point cloud
- `reconstructed_mesh.ply`: 3D surface mesh
- `reconstruction_log.txt`: Full execution log with exit codes
- `colmap_status.txt`: Diagnostic guide if COLMAP is missing.

---

## 5. Verification Summary Table

| Question | Script | Primary CV Methods | Output Directory | Verification Status |
|:---|:---|:---|:---|:---|
| **Q1(a)** | `q1a_2d_transformations.py` | `warpAffine`, `getRotationMatrix2D` | `output/q1a/` | **Verified & Generated** |
| **Q1(b)** | `q1b_3d_transformations.py` | 3D Rotations ($R_x, R_y, R_z$), Translation, `mplot3d` | `output/q1b/` | **Verified & Generated** |
| **Q1(c)** | `q1c_homogeneous_coordinates.py` | Homogeneous formulation, $3 \times 3$ matrices | `output/q1c/` | **Verified & Generated** |
| **Q1(d)** | `q1d_affine_transformations.py` | `getAffineTransform`, collinearity verification | `output/q1d/` | **Verified & Generated** |
| **Q1(e)** | `q1e_projective_homography.py` | `findHomography`, `warpPerspective` | `output/q1e/` | **Verified & Generated** |
| **Q2** | `q2_perspective_correction.py` | `getPerspectiveTransform` (Tilted $\rightarrow$ Rect) | `output/q2/` | **Verified & Generated** |
| **Q3** | `q3_undistort_live_camera.py` | `calibrateCamera`, `VideoCapture`, `undistort` | `output/q3/` | **Verified & Generated** |
| **Q4** | `q4_colmap_3d_reconstruction.py` | SfM, MVS, Poisson/Delaunay meshing | `output/q4/` | **Verified & Diagnostic Tested** |


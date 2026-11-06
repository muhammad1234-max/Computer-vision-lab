"""
Q4: Build a 3D Model from Images Using COLMAP
Automates the full 3D reconstruction pipeline:
1. Feature detection and extraction
2. Feature matching across images
3. Camera pose estimation and sparse reconstruction (SfM)
4. Dense reconstruction (stereo fusion -> dense point cloud)
5. 3D surface mesh generation (Poisson / Delaunay mesher)
"""

import os
import sys
import shutil
import argparse
import subprocess
import glob


def find_colmap_executable(user_path=None):
    """
    Locates the COLMAP executable on the system.
    Searches user-specified path, system PATH, and standard Windows directories.
    """
    if user_path:
        if os.path.isfile(user_path) and os.access(user_path, os.X_OK):
            return user_path
        # Check if user passed directory containing colmap
        for candidate in ["colmap.bat", "COLMAP.bat", "colmap.exe"]:
            p = os.path.join(user_path, candidate)
            if os.path.isfile(p):
                return p

    # Check system PATH
    for name in ["colmap.bat", "COLMAP.bat", "colmap.exe", "colmap"]:
        found = shutil.which(name)
        if found:
            return found

    # Check common Windows install directories
    candidate_paths = [
        r"C:\colmap\colmap.bat",
        r"C:\colmap\COLMAP.bat",
        r"C:\colmap\colmap.exe",
        r"C:\Program Files\COLMAP\colmap.bat",
        r"C:\Program Files\COLMAP\COLMAP.bat",
        r"C:\Program Files\COLMAP\colmap.exe",
        r"C:\Program Files (x86)\COLMAP\colmap.bat",
        r"C:\Program Files (x86)\COLMAP\COLMAP.bat",
        r"C:\Program Files (x86)\COLMAP\colmap.exe",
        os.path.expanduser(r"~\COLMAP\colmap.bat"),
        os.path.expanduser(r"~\COLMAP\colmap.exe"),
        os.path.expanduser(r"~\Downloads\COLMAP\colmap.bat"),
        os.path.expanduser(r"~\Downloads\COLMAP\colmap.exe"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    return None


def run_command(cmd, log_file, stage_name=""):
    """
    Executes a shell command, logs output to file and console,
    and checks return code.
    """
    print(f"\n{'='*70}")
    print(f"[Q4] Executing Stage: {stage_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n\n{'='*70}\nSTAGE: {stage_name}\nCOMMAND: {' '.join(cmd)}\n{'='*70}\n")

    use_shell = sys.platform.startswith("win") and cmd[0].lower().endswith(".bat")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        shell=use_shell
    )

    full_output = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        full_output.append(line)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(line)

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        error_msg = f"[Q4] ERROR: Stage '{stage_name}' failed with exit code {return_code}."
        print(error_msg, file=sys.stderr)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n{error_msg}\n")
        return False, "".join(full_output)

    print(f"[Q4] Stage '{stage_name}' completed successfully.")
    return True, "".join(full_output)


def validate_input_images(image_dir):
    """Verifies that the image directory exists and contains valid image files."""
    if not os.path.isdir(image_dir):
        return False, f"Directory '{image_dir}' does not exist."

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))

    if len(images) < 3:
        return False, f"Directory '{image_dir}' contains {len(images)} images. COLMAP requires at least 3 images for triangulation."

    return True, images


def write_colmap_instructions(output_dir, searched_locations):
    """
    Generates a clear setup guide when COLMAP is not detected on the machine.
    """
    guide = f"""COLMAP 3D RECONSTRUCTION ENVIRONMENT DIAGNOSTIC (Lab 07 - Question 4)
=============================================================================

STATUS:
COLMAP executable was not found on this system.

LOCATIONS SEARCHED:
{chr(10).join(f"- {loc}" for loc in searched_locations)}

TO RUN THE COMPLETE 3D RECONSTRUCTION PIPELINE:
------------------------------------------------
1. Download pre-built COLMAP for Windows:
   Visit: https://github.com/colmap/colmap/releases
   Download either:
     - colmap-x64-windows-cuda.zip    (Recommended if NVIDIA GPU with CUDA is available)
     - colmap-x64-windows-no-cuda.zip (CPU-only version)

2. Extract the archive:
   Extract to: C:\\colmap
   (You should have C:\\colmap\\colmap.bat or C:\\colmap\\COLMAP.bat)

3. Add to System PATH or run with --colmap_path:
   python q4_colmap_3d_reconstruction.py --colmap_path "C:\\colmap\\colmap.bat"

PIPELINE AUTOMATION SUMMARY:
When COLMAP is installed, q4_colmap_3d_reconstruction.py automatically executes:
  Stage 1: colmap feature_extractor (SIFT features stored in output/q4/database.db)
  Stage 2: colmap exhaustive_matcher (Matches features across image pairs)
  Stage 3: colmap mapper (Estimates camera poses and computes sparse 3D point cloud)
  Stage 4: colmap image_undistorter + patch_match_stereo + stereo_fusion (Dense point cloud .ply)
  Stage 5: colmap poisson_mesher / delaunay_mesher (Generates 3D mesh .ply)

All outputs will be saved directly into: output/q4/
"""
    status_file = os.path.join(output_dir, "colmap_status.txt")
    with open(status_file, "w", encoding="utf-8") as f:
        f.write(guide)
    return status_file, guide


def main():
    parser = argparse.ArgumentParser(description="Q4 - COLMAP 3D Reconstruction Pipeline")
    parser.add_argument("--image_dir", type=str, default="input/colmap_images",
                        help="Directory containing input photographs (default: input/colmap_images)")
    parser.add_argument("--colmap_path", type=str, default=None,
                        help="Explicit path to colmap.bat or colmap.exe")
    parser.add_argument("--output_dir", type=str, default="output/q4",
                        help="Output directory for reconstruction data (default: output/q4)")
    parser.add_argument("--use_gpu", type=int, default=0,
                        help="Enable GPU/CUDA: 1 for GPU, 0 for CPU (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "reconstruction_log.txt")

    print("=" * 70)
    print("   LAB 07 - QUESTION 4: 3D RECONSTRUCTION USING COLMAP")
    print("=" * 70)

    # 1. Validate Input Images
    valid, img_result = validate_input_images(args.image_dir)
    if not valid:
        print(f"\n[Q4] Input Validation Error: {img_result}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Q4] Input image directory: '{args.image_dir}' ({len(img_result)} images found)")
    for img_p in img_result[:5]:
        print(f"  - {os.path.basename(img_p)}")
    if len(img_result) > 5:
        print(f"  ... and {len(img_result) - 5} more.")

    # 2. Check COLMAP Executable
    colmap_exec = find_colmap_executable(args.colmap_path)
    if not colmap_exec:
        searched = [
            args.colmap_path if args.colmap_path else "(No --colmap_path provided)",
            "System PATH (colmap, colmap.bat, colmap.exe)",
            r"C:\colmap",
            r"C:\Program Files\COLMAP",
            os.path.expanduser(r"~\COLMAP"),
        ]
        status_file, guide = write_colmap_instructions(args.output_dir, searched)
        print(f"\n{'!'*70}")
        print(guide)
        print(f"{'!'*70}")
        print(f"\n[Q4] Diagnostic report written to: '{status_file}'")
        return

    print(f"\n[Q4] Found COLMAP executable: '{colmap_exec}'")

    # Inspect COLMAP version/help
    ver_ok, ver_out = run_command([colmap_exec, "-h"], log_file, "COLMAP Version & Capability Check")
    if not ver_ok:
        print("[Q4] Warning: Failed to retrieve COLMAP help text. Continuing with standard options...")

    database_path = os.path.join(args.output_dir, "database.db")
    sparse_path = os.path.join(args.output_dir, "sparse")
    dense_path = os.path.join(args.output_dir, "dense")

    # Clean previous run intermediate files to ensure clean pipeline execution
    if os.path.exists(dense_path):
        shutil.rmtree(dense_path, ignore_errors=True)
    if os.path.exists(sparse_path):
        shutil.rmtree(sparse_path, ignore_errors=True)
    if os.path.exists(database_path):
        try:
            os.remove(database_path)
        except OSError:
            pass

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(dense_path, exist_ok=True)

    # Stage 1: Feature Extraction
    _, ext_help = run_command([colmap_exec, "feature_extractor", "-h"], log_file, "Inspect feature_extractor Options")
    cmd_extract = [
        colmap_exec, "feature_extractor",
        "--database_path", database_path,
        "--image_path", args.image_dir,
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1"
    ]
    if "--FeatureExtraction.use_gpu" in ext_help:
        cmd_extract.extend(["--FeatureExtraction.use_gpu", str(args.use_gpu)])
    elif "--SiftExtraction.use_gpu" in ext_help:
        cmd_extract.extend(["--SiftExtraction.use_gpu", str(args.use_gpu)])

    ok, _ = run_command(cmd_extract, log_file, "1. Feature Extraction (SIFT)")
    if not ok:
        print("[Q4] Feature extraction failed. Check reconstruction_log.txt for details.")
        return

    # Stage 2: Feature Matching
    _, match_help = run_command([colmap_exec, "exhaustive_matcher", "-h"], log_file, "Inspect exhaustive_matcher Options")
    cmd_match = [
        colmap_exec, "exhaustive_matcher",
        "--database_path", database_path
    ]
    if "--FeatureMatching.use_gpu" in match_help:
        cmd_match.extend(["--FeatureMatching.use_gpu", str(args.use_gpu)])
    elif "--SiftMatching.use_gpu" in match_help:
        cmd_match.extend(["--SiftMatching.use_gpu", str(args.use_gpu)])

    ok, _ = run_command(cmd_match, log_file, "2. Feature Matching (Exhaustive)")
    if not ok:
        print("[Q4] Feature matching failed. Check reconstruction_log.txt for details.")
        return

    # Stage 3: Sparse Reconstruction (SfM / Camera Pose Estimation)
    cmd_mapper = [
        colmap_exec, "mapper",
        "--database_path", database_path,
        "--image_path", args.image_dir,
        "--output_path", sparse_path
    ]
    ok, _ = run_command(cmd_mapper, log_file, "3. Sparse Reconstruction & Camera Pose Estimation")
    if not ok:
        print("[Q4] Sparse mapper failed. Check reconstruction_log.txt for details.")
        return

    # Export sparse model to PLY
    sparse_model_dir = os.path.join(sparse_path, "0")
    sparse_ply = os.path.join(args.output_dir, "sparse_point_cloud.ply")
    if os.path.isdir(sparse_model_dir):
        cmd_convert = [
            colmap_exec, "model_converter",
            "--input_path", sparse_model_dir,
            "--output_path", sparse_ply,
            "--output_type", "PLY"
        ]
        run_command(cmd_convert, log_file, "Export Sparse Point Cloud (.ply)")

    # Stage 4: Dense Reconstruction (Dense Point Cloud)
    cmd_undistort = [
        colmap_exec, "image_undistorter",
        "--image_path", args.image_dir,
        "--input_path", sparse_model_dir,
        "--output_path", dense_path
    ]
    ok_undist, _ = run_command(cmd_undistort, log_file, "4a. Image Undistortion for Dense MVS")

    fused_ply = os.path.join(dense_path, "fused.ply")
    if ok_undist:
        cmd_stereo = [
            colmap_exec, "patch_match_stereo",
            "--workspace_path", dense_path,
            "--PatchMatchStereo.geom_consistency", "true"
        ]
        ok_stereo, _ = run_command(cmd_stereo, log_file, "4b. PatchMatch Stereo Depth Estimation")

        if ok_stereo:
            cmd_fusion = [
                colmap_exec, "stereo_fusion",
                "--workspace_path", dense_path,
                "--output_path", fused_ply
            ]
            ok_fusion, _ = run_command(cmd_fusion, log_file, "4c. Stereo Fusion (Dense Point Cloud)")
            if ok_fusion and os.path.isfile(fused_ply):
                shutil.copyfile(fused_ply, os.path.join(args.output_dir, "dense_point_cloud.ply"))
                print(f"[Q4] Dense point cloud generated: {fused_ply}")

    # Stage 5: Mesh Reconstruction (Point Cloud -> 3D Mesh)
    mesh_ply = os.path.join(args.output_dir, "reconstructed_mesh.ply")
    input_cloud = fused_ply if os.path.isfile(fused_ply) else sparse_ply

    if os.path.isfile(input_cloud):
        # Try Poisson Mesher
        cmd_poisson = [
            colmap_exec, "poisson_mesher",
            "--input_path", input_cloud,
            "--output_path", mesh_ply
        ]
        ok_mesh, _ = run_command(cmd_poisson, log_file, "5. Surface Meshing (Poisson)")

        # If Poisson fails or not supported, try Delaunay mesher
        if not ok_mesh:
            use_dense = os.path.isdir(dense_path) and os.path.isfile(fused_ply)
            in_p = dense_path if use_dense else sparse_model_dir
            in_type = "dense" if use_dense else "sparse"
            if os.path.exists(in_p):
                cmd_delaunay = [
                    colmap_exec, "delaunay_mesher",
                    "--input_path", in_p,
                    "--input_type", in_type,
                    "--output_path", mesh_ply
                ]
                run_command(cmd_delaunay, log_file, f"5 (Fallback). Surface Meshing (Delaunay {in_type})")

    print(f"\n[Q4] Reconstruction pipeline finished. All logs in '{log_file}'.")


if __name__ == "__main__":
    main()


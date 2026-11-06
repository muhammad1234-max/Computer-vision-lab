"""
Q3: Undistort Live Camera Feed
Implements:
1. Camera calibration using a checkerboard pattern.
2. Saving camera matrix and distortion coefficients to output/q3/camera_calibration.npz.
3. Live webcam feed via cv2.VideoCapture() with real-time cv2.undistort().
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np


def calibrate_from_images(image_list, pattern_size=(9, 6), square_size=25.0):
    """
    Calibrates camera from a list of images containing a checkerboard pattern.
    pattern_size: (columns, rows) of INNER corners.
    square_size: physical square dimension (e.g. in millimeters).
    """
    cols, rows = pattern_size
    # Prepare 3D object points like (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    valid_frames = 0

    print(f"\n[Q3] Processing calibration frames for {cols}x{rows} inner corners...")

    for idx, img in enumerate(image_list):
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if img_shape is None:
            img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(refined_corners)
            valid_frames += 1
            print(f"  Frame {idx + 1:02d}: Checkerboard detected and corners refined.")
        else:
            print(f"  Frame {idx + 1:02d}: Checkerboard NOT found.")

    if valid_frames < 3:
        raise ValueError(f"Need at least 3 valid calibration frames with detected checkerboards. Got {valid_frames}.")

    print(f"[Q3] Calibrating camera using {valid_frames} valid views...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Compute re-projection error
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += err * err
        total_points += len(objpoints[i])
    mean_reproj_error = np.sqrt(total_error / total_points)

    print(f"[Q3] Calibration complete! Mean re-projection RMS error: {ret:.4f} px (mean point error: {mean_reproj_error:.4f} px)")
    return ret, mtx, dist, img_shape


def save_calibration(output_dir, mtx, dist, img_shape=None, error=None):
    """Save calibration parameters to reusable npz and human-readable txt."""
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "camera_calibration.npz")
    np.savez(npz_path, camera_matrix=mtx, dist_coeffs=dist, img_shape=img_shape)

    txt_path = os.path.join(output_dir, "camera_calibration.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CAMERA CALIBRATION PARAMETERS (Lab 07 - Question 3)\n")
        f.write("=" * 55 + "\n\n")
        f.write("1. INTRINSIC CAMERA MATRIX (K) [3x3]:\n")
        f.write(f"   fx = {mtx[0,0]:.4f}, fy = {mtx[1,1]:.4f}\n")
        f.write(f"   cx = {mtx[0,2]:.4f}, cy = {mtx[1,2]:.4f}\n")
        f.write(f"{mtx}\n\n")
        f.write("2. DISTORTION COEFFICIENTS [k1, k2, p1, p2, k3]:\n")
        f.write(f"{dist.ravel()}\n\n")
        if img_shape:
            f.write(f"Calibrated Image Resolution: {img_shape[0]} x {img_shape[1]}\n")
        if error:
            f.write(f"RMS Re-projection Error: {error:.4f} pixels\n")

    print(f"[Q3] Saved reusable calibration files:")
    print(f"  - Binary: {npz_path}")
    print(f"  - Readable: {txt_path}")
    return npz_path


def load_calibration(output_dir):
    """Load saved calibration data from npz."""
    npz_path = os.path.join(output_dir, "camera_calibration.npz")
    if not os.path.exists(npz_path):
        return None, None, None
    data = np.load(npz_path)
    mtx = data["camera_matrix"]
    dist = data["dist_coeffs"]
    img_shape = data["img_shape"] if "img_shape" in data else None
    return mtx, dist, img_shape


def generate_synthetic_calibration_views(checkerboard_path, num_views=8):
    """
    Creates multiple perspective warped views of a checkerboard pattern
    to enable fully automated calibration testing without a physical printed board.
    """
    board = cv2.imread(checkerboard_path)
    if board is None:
        raise FileNotFoundError(f"Checkerboard image not found at '{checkerboard_path}'.")

    h, w = board.shape[:2]
    views = []
    # Base corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    rng = np.random.RandomState(42)
    for i in range(num_views):
        # Apply realistic 3D tilt
        dx1, dy1 = rng.uniform(-40, 40), rng.uniform(-30, 30)
        dx2, dy2 = rng.uniform(-40, 40), rng.uniform(-30, 30)
        dx3, dy3 = rng.uniform(-40, 40), rng.uniform(-30, 30)
        dx4, dy4 = rng.uniform(-40, 40), rng.uniform(-30, 30)

        dst = np.float32([
            [60 + dx1, 60 + dy1],
            [w - 60 + dx2, 50 + dy2],
            [w - 40 + dx3, h - 50 + dy3],
            [50 + dx4, h - 60 + dy4]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(board, M, (w, h), borderValue=(255, 255, 255))
        views.append(warped)

    return views


def run_interactive_calibration(camera_id=0, pattern_size=(9, 6), square_size=25.0):
    """
    Captures live frames from webcam for checkerboard calibration.
    Press 'c' to capture a calibration frame when checkerboard is visible.
    Press 's' to finish and calibrate.
    Press 'q' to abort.
    """
    cols, rows = pattern_size
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_id}.")

    print("\n--- Live Camera Calibration Instructions ---")
    print("1. Point your webcam at a printed checkerboard.")
    print("2. Press 'c' to capture a calibration snapshot when corners are highlighted.")
    print("3. Capture at least 10-15 snapshots from different angles and distances.")
    print("4. Press 's' to compute and save calibration.")
    print("5. Press 'q' to cancel.")

    captured_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            status_text = f"FOUND! Press 'c' to capture ({len(captured_frames)} captured)"
            color = (0, 255, 0)
        else:
            status_text = f"Looking for {cols}x{rows} board... ({len(captured_frames)} captured)"
            color = (0, 0, 255)

        cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "[c] Capture  [s] Calibrate  [q] Quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Webcam Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and found:
            captured_frames.append(frame.copy())
            print(f"Captured calibration frame {len(captured_frames)}")
        elif key == ord('s'):
            if len(captured_frames) >= 3:
                print(f"Proceeding to calibrate with {len(captured_frames)} frames...")
                break
            else:
                print(f"Need at least 3 frames (currently have {len(captured_frames)}).")
        elif key == ord('q'):
            print("Calibration aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, shape = calibrate_from_images(captured_frames, pattern_size, square_size)
    return ret, mtx, dist, shape


def run_live_undistort(camera_id, mtx, dist, output_dir, max_frames=None):
    """
    Opens cv2.VideoCapture() and applies cv2.undistort() in real time.
    Displays original and undistorted feeds side-by-side.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_id}.")

    print("\n--- Live Camera Undistortion Running ---")
    print("Side-by-side feed active: Left = Original (Distorted), Right = Undistorted (cv2.undistort)")
    print("Controls:")
    print("  's' - Save snapshot of current frames to output/q3/")
    print("  'q' or ESC - Exit camera feed cleanly")

    frame_count = 0
    saved_snapshot = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame from camera stream.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Apply cv2.undistort as required by the specification
        undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

        # Label feeds
        display_orig = frame.copy()
        display_undist = undistorted.copy()

        cv2.putText(display_orig, "ORIGINAL (DISTORTED)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_undist, "UNDISTORTED (cv2.undistort)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Combine side-by-side
        combined = np.hstack([display_orig, display_undist])

        # Add overall status overlay
        cv2.putText(combined, "Press 's' to snapshot | 'q' to quit", (w - 180, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # In headless or test mode with max_frames
        if max_frames is not None:
            if not saved_snapshot:
                sample_path = os.path.join(output_dir, "live_undistort_sample.png")
                cv2.imwrite(sample_path, combined)
                saved_snapshot = True
                print(f"[Q3] Saved live undistorted frame snapshot to '{sample_path}'")
            if frame_count >= max_frames:
                break
        else:
            cv2.imshow("Lab 07 Q3: Live Webcam Undistortion Comparison", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                sample_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.png")
                cv2.imwrite(sample_path, combined)
                print(f"[Q3] Saved snapshot to '{sample_path}'")
            elif key == ord('q') or key == 27:  # 'q' or ESC
                print("[Q3] User requested exit. Closing feed.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[Q3] Camera released and OpenCV windows closed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Q3 - Undistort Live Camera Feed")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive camera calibration using live webcam")
    parser.add_argument("--simulate_calibration", action="store_true",
                        help="Perform calibration using synthetic/provided checkerboard images")
    parser.add_argument("--cols", type=int, default=9,
                        help="Number of inner corners horizontally (default 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="Number of inner corners vertically (default 6)")
    parser.add_argument("--square_size", type=float, default=25.0,
                        help="Checkerboard square size in mm (default 25.0)")
    parser.add_argument("--test", action="store_true",
                        help="Run a verification test (capture/process frames and exit)")
    parser.add_argument("--output_dir", type=str, default="output/q3",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pattern_size = (args.cols, args.rows)

    # Check for existing calibration data
    mtx, dist, shape = load_calibration(args.output_dir)

    # Mode 1: Interactive live calibration
    if args.calibrate:
        print("[Q3] Initiating interactive webcam calibration...")
        res = run_interactive_calibration(args.camera, pattern_size, args.square_size)
        if res is not None:
            ret, mtx, dist, shape = res
            save_calibration(args.output_dir, mtx, dist, shape, ret)
        return

    # Mode 2: Automated / Simulated calibration from checkerboard
    if args.simulate_calibration or mtx is None:
        if mtx is None and not args.simulate_calibration and not args.test:
            print("[Q3] Notice: No saved calibration file found in 'output/q3/'.")
            print("     Automatically calibrating using reference checkerboard patterns...")

        checkerboard_file = "input/checkerboard.png"
        if not os.path.exists(checkerboard_file):
            raise FileNotFoundError(f"Missing '{checkerboard_file}'. Please generate or provide a calibration board.")

        views = generate_synthetic_calibration_views(checkerboard_file, num_views=8)
        ret, mtx, dist, shape = calibrate_from_images(views, pattern_size, args.square_size)
        save_calibration(args.output_dir, mtx, dist, shape, ret)

    print("\n--- Loaded Calibration Data ---")
    print("Camera Intrinsic Matrix (K):")
    print(mtx)
    print("Distortion Coefficients:")
    print(dist.ravel())

    # Mode 3: Live feed with cv2.undistort()
    max_frames = 15 if args.test else None
    run_live_undistort(args.camera, mtx, dist, args.output_dir, max_frames=max_frames)


if __name__ == "__main__":
    main()


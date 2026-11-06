"""
Q1(c) Homogeneous Coordinates
Demonstrates the representation of 2D points in homogeneous coordinates,
explains why the extra dimension is vital in Computer Vision, and shows
how translation becomes a linear matrix multiplication.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def cartesian_to_homogeneous(pt):
    """Convert a 2D Cartesian point (x, y) to homogeneous coordinates [x, y, 1]."""
    x, y = pt
    return np.array([x, y, 1.0], dtype=np.float64)


def homogeneous_to_cartesian(pt_h):
    """
    Convert a homogeneous coordinate [x, y, w] back to Cartesian (x/w, y/w).
    Raises ValueError if w == 0 (point at infinity).
    """
    w = pt_h[2]
    if np.isclose(w, 0.0):
        raise ValueError("Cannot convert ideal point at infinity (w=0) to Euclidean Cartesian coordinates.")
    return np.array([pt_h[0] / w, pt_h[1] / w], dtype=np.float64)


def build_homogeneous_translation_matrix(tx, ty):
    """Construct 3x3 translation matrix in homogeneous coordinates."""
    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def build_homogeneous_rotation_matrix(angle_deg):
    """Construct 3x3 rotation matrix around origin in homogeneous coordinates."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Q1(c) - Homogeneous Coordinates")
    parser.add_argument("--output_dir", type=str, default="output/q1c",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Define a 2D Cartesian point
    p_cartesian = np.array([4.0, 2.5], dtype=np.float64)
    print("=================================================================")
    print("   Q1(c): 2D POINT IN HOMOGENEOUS COORDINATES & MATRIX MAPPING   ")
    print("=================================================================")
    print(f"\n1. Original 2D Cartesian point: (x, y) = ({p_cartesian[0]}, {p_cartesian[1]})")

    # 2. Convert to Homogeneous Coordinates
    p_homo = cartesian_to_homogeneous(p_cartesian)
    print(f"2. Homogeneous representation: [x, y, w]^T = [{p_homo[0]}, {p_homo[1]}, {p_homo[2]}]^T")
    print("   Notice the extra coordinate w = 1.0.")
    print("   In projective space, any scaled vector [k*x, k*y, k]^T represents the exact same physical point.")

    # Show projective equivalence
    k = 3.5
    p_homo_scaled = p_homo * k
    print(f"   Example scaled vector (k={k}): [{p_homo_scaled[0]}, {p_homo_scaled[1]}, {p_homo_scaled[2]}]^T")
    p_recovered = homogeneous_to_cartesian(p_homo_scaled)
    print(f"   Converted back to Cartesian: ({p_recovered[0]}, {p_recovered[1]}) -> Identical!")

    # 3. Demonstration of Translation via Matrix Multiplication
    tx, ty = 5.0, 3.0
    T = build_homogeneous_translation_matrix(tx, ty)
    p_translated_homo = T @ p_homo
    p_translated_cart = homogeneous_to_cartesian(p_translated_homo)

    print(f"\n3. Translation Demonstration (tx={tx}, ty={ty}):")
    print("   3x3 Homogeneous Translation Matrix T:")
    print(T)
    print(f"   Matrix-vector product T @ [x, y, 1]^T = [{p_translated_homo[0]}, {p_translated_homo[1]}, {p_translated_homo[2]}]^T")
    print(f"   Resulting 2D Cartesian Point: ({p_translated_cart[0]}, {p_translated_cart[1]})")

    # 4. Chained Transformation (Rotation followed by Translation)
    R = build_homogeneous_rotation_matrix(45)
    M_combined = T @ R  # Rotate then translate
    p_transformed_homo = M_combined @ p_homo
    p_transformed_cart = homogeneous_to_cartesian(p_transformed_homo)

    print("\n4. Transformation Composition (Rotation 45 deg + Translation):")
    print("   Combined Matrix M = T @ R:")
    print(np.round(M_combined, 4))
    print(f"   Resulting Cartesian Point: ({p_transformed_cart[0]:.4f}, {p_transformed_cart[1]:.4f})")

    # 5. Comprehensive Explanation Text
    explanation_text = f"""LAB 07 - QUESTION 1(c): HOMOGENEOUS COORDINATES IN COMPUTER VISION

1. DEFINITION & FORMULATION
---------------------------
A 2D Euclidean Cartesian point is defined as:
    P_cartesian = (x, y)

In homogeneous (projective) coordinates, this point is represented as a 3D ray:
    P_homo = [X, Y, W]^T  such that x = X / W and y = Y / W (for W != 0)

Conventionally, finite Euclidean points use W = 1:
    Original Point:       (x, y) = ({p_cartesian[0]}, {p_cartesian[1]})
    Homogeneous Vector:   [x, y, 1]^T = [{p_homo[0]}, {p_homo[1]}, {p_homo[2]}]^T

Any non-zero scalar multiple k * [x, y, 1]^T represents the exact same physical point in Cartesian space:
    [{p_homo_scaled[0]}, {p_homo_scaled[1]}, {p_homo_scaled[2]}]^T ---> ({p_recovered[0]}, {p_recovered[1]})


2. WHY IS THE EXTRA COORDINATE (W) USED IN COMPUTER VISION?
------------------------------------------------------------
(a) LINEAR REPRESENTATION OF TRANSLATION:
    In standard 2D Cartesian coordinates, rotation and scaling are linear transformations:
        x' = R * x
    However, translation is an AFFINE operation requiring vector addition:
        x' = x + t
    This addition prevents translation from being combined directly with linear transformations into a single 2x2 matrix.
    By augmenting the coordinate vector with an extra component (W=1), translation becomes a linear matrix multiplication:
        [x']   [1  0  tx] [x]
        [y'] = [0  1  ty] [y]
        [ 1]   [0  0   1] [1]

(b) UNIFIED TRANSFORMATION COMPOSITION:
    In Computer Vision pipelines (e.g., camera calibration, robotics, SLAM), multiple operations (rotation, scaling,
    shearing, translation, and projection) must be chained repeatedly.
    With homogeneous coordinates, any sequence of transformations can be multiplied together into a single 3x3 matrix:
        M = T * R * S
    A single matrix-vector multiplication then transforms any point, dramatically boosting computational efficiency.

(c) PERSPECTIVE PROJECTION & CAMERAS:
    Pinhole cameras project 3D world points (X, Y, Z) onto a 2D image sensor (u, v) by dividing by depth Z:
        u = f * (X / Z) + cx
        v = f * (Y / Z) + cy
    Division is non-linear in Euclidean space. In homogeneous coordinates, projection is represented linearly by a 3x4
    camera projection matrix P:
        [u*w, v*w, w]^T = P * [X, Y, Z, 1]^T
    The division by depth (perspective foreshortening) occurs naturally during homogeneous normalization by W.

(d) REPRESENTATION OF POINTS AT INFINITY:
    Parallel lines in the real world (such as railway tracks) intersect at a vanishing point. In Euclidean space, this
    intersection occurs at infinity and cannot be assigned finite coordinates.
    In homogeneous coordinates, points at infinity are represented naturally when W = 0: [dx, dy, 0]^T, encoding the
    direction of the line without causing a numerical singularity.


3. NUMERICAL VERIFICATION
-------------------------
Initial Point:
    P = ({p_cartesian[0]}, {p_cartesian[1]})
Translation Vector:
    (tx, ty) = ({tx}, {ty})
Homogeneous Matrix Multiplication:
    T @ P_homo = [{p_translated_homo[0]}, {p_translated_homo[1]}, {p_translated_homo[2]}]^T
Cartesian Extraction:
    P_translated = ({p_translated_cart[0]}, {p_translated_cart[1]})
"""

    # Print explanation
    print("\n" + "="*65)
    print(explanation_text)
    print("="*65)

    # Save explanation text to file
    txt_path = os.path.join(args.output_dir, "homogeneous_coordinates_explanation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(explanation_text)
    print(f"\n[Q1c] Saved explanation to '{txt_path}'")

    # 6. Generate an intuitive visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: 2D Translation in Cartesian Plane
    ax1.plot([p_cartesian[0]], [p_cartesian[1]], "bo", markersize=10, label=f"Original P({p_cartesian[0]}, {p_cartesian[1]})")
    ax1.plot([p_translated_cart[0]], [p_translated_cart[1]], "ro", markersize=10,
             label=f"Translated P'({p_translated_cart[0]}, {p_translated_cart[1]}) via T @ P_h")
    ax1.annotate("", xy=p_translated_cart, xytext=p_cartesian,
                 arrowprops=dict(facecolor="green", edgecolor="green", width=2, headwidth=10))
    ax1.text((p_cartesian[0] + p_translated_cart[0])/2 - 1.0,
             (p_cartesian[1] + p_translated_cart[1])/2 + 0.4,
             f"+(tx={tx}, ty={ty})", fontsize=11, color="darkgreen", fontweight="bold")
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 8)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("2D Cartesian View: Translation as Vector Shift", fontweight="bold")
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.legend()

    # Plot 2: Projective Space Representation (3D Ray through W=1 plane)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Draw W=1 plane
    wx, wy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 8, 10))
    ww = np.ones_like(wx)
    ax2.plot_surface(wx, wy, ww, alpha=0.15, color="cyan")

    # Rays from origin through [x, y, 1]
    ax2.plot([0, p_homo[0]*2], [0, p_homo[1]*2], [0, 2], "b--", label="Projective Ray (Original)")
    ax2.scatter([p_homo[0]], [p_homo[1]], [1], color="blue", s=60)
    ax2.text(p_homo[0], p_homo[1], 1.1, " [x, y, 1]", color="blue")

    ax2.plot([0, p_translated_homo[0]*1.4], [0, p_translated_homo[1]*1.4], [0, 1.4], "r--", label="Projective Ray (Translated)")
    ax2.scatter([p_translated_homo[0]], [p_translated_homo[1]], [1], color="red", s=60)
    ax2.text(p_translated_homo[0], p_translated_homo[1], 1.1, " [x+tx, y+ty, 1]", color="red")

    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 8)
    ax2.set_zlim(0, 2)
    ax2.set_xlabel("X", fontweight="bold")
    ax2.set_ylabel("Y", fontweight="bold")
    ax2.set_zlabel("W (Scale)", fontweight="bold")
    ax2.set_title("Projective Space: Intersection with W=1 Plane", fontweight="bold")
    ax2.legend()
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "homogeneous_translation_demo.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Q1c] Saved visualization diagram to '{plot_path}'")


if __name__ == "__main__":
    main()


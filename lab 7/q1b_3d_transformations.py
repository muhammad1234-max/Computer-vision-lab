"""
Q1(b) 3D Transformations
Demonstrates 3D geometric transformations (rotation around X, Y, Z axes
and translation along X, Y, Z axes) on a 3D cube object, visualizing the results.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_cube(size=1.0):
    """
    Generate 8 vertices of a cube centered at the origin,
    along with its 12 edges and 6 faces.
    """
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s]
    ], dtype=np.float64)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical pillars
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]

    return vertices, edges, faces


def get_rotation_matrix_x(angle_deg):
    """3D rotation matrix around X-axis."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])


def get_rotation_matrix_y(angle_deg):
    """3D rotation matrix around Y-axis."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def get_rotation_matrix_z(angle_deg):
    """3D rotation matrix around Z-axis."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def apply_3d_transform(vertices, R=None, t=None):
    """Apply 3D rotation R (3x3) and translation t (3,) to Nx3 vertices."""
    transformed = vertices.copy()
    if R is not None:
        # V' = V @ R.T  (since vertices are row vectors: v' = R @ v)
        transformed = transformed @ R.T
    if t is not None:
        transformed = transformed + np.array(t, dtype=np.float64)
    return transformed


def plot_cube(ax, vertices, edges, color="blue", alpha=0.3, label="Object"):
    """Plot cube wireframe edges and translucent faces."""
    # Plot edges
    for i, (start, end) in enumerate(edges):
        p1, p2 = vertices[start], vertices[end]
        ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                  color=color, linewidth=2,
                  label=label if i == 0 else "")

    # Plot faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)


def create_3d_figure(orig_vertices, trans_vertices, edges, title,
                     save_path=None, limits=(-3, 3)):
    """Generate a single comparison 3D plot."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original in muted gray/blue
    plot_cube(ax, orig_vertices, edges, color="royalblue", alpha=0.2, label="Original (at origin)")
    # Plot transformed in vibrant orange/red
    plot_cube(ax, trans_vertices, edges, color="crimson", alpha=0.4, label="Transformed")

    # Draw coordinate axes lines through origin
    ax.plot3D([limits[0], limits[1]], [0, 0], [0, 0], "k--", linewidth=0.8, alpha=0.5)
    ax.plot3D([0, 0], [limits[0], limits[1]], [0, 0], "k--", linewidth=0.8, alpha=0.5)
    ax.plot3D([0, 0], [0, 0], [limits[0], limits[1]], "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    ax.set_xlabel("X axis", fontweight="bold")
    ax.set_ylabel("Y axis", fontweight="bold")
    ax.set_zlabel("Z axis", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=45)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Q1(b) - 3D Transformations")
    parser.add_argument("--output_dir", type=str, default="output/q1b",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vertices, edges, _ = create_cube(size=1.5)

    print("[Q1b] Initial 3D Cube Vertices (Centered at Origin):")
    for i, v in enumerate(vertices):
        print(f"  V{i}: [{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}]")

    # 1. Rotation around X-axis (45 degrees)
    Rx = get_rotation_matrix_x(45)
    v_rot_x = apply_3d_transform(vertices, R=Rx)
    create_3d_figure(vertices, v_rot_x, edges,
                     "3D Transformation: Rotation around X-axis (45°)",
                     os.path.join(args.output_dir, "1_rotation_x.png"))
    print("\n[Q1b] Computed Rotation around X-axis (45 deg)")

    # 2. Rotation around Y-axis (45 degrees)
    Ry = get_rotation_matrix_y(45)
    v_rot_y = apply_3d_transform(vertices, R=Ry)
    create_3d_figure(vertices, v_rot_y, edges,
                     "3D Transformation: Rotation around Y-axis (45°)",
                     os.path.join(args.output_dir, "2_rotation_y.png"))
    print("[Q1b] Computed Rotation around Y-axis (45 deg)")

    # 3. Rotation around Z-axis (45 degrees)
    Rz = get_rotation_matrix_z(45)
    v_rot_z = apply_3d_transform(vertices, R=Rz)
    create_3d_figure(vertices, v_rot_z, edges,
                     "3D Transformation: Rotation around Z-axis (45°)",
                     os.path.join(args.output_dir, "3_rotation_z.png"))
    print("[Q1b] Computed Rotation around Z-axis (45 deg)")

    # 4. Translation along X, Y, Z axes
    t_xyz = [1.5, 1.0, 1.2]
    v_trans = apply_3d_transform(vertices, t=t_xyz)
    create_3d_figure(vertices, v_trans, edges,
                     f"3D Transformation: Translation (tx={t_xyz[0]}, ty={t_xyz[1]}, tz={t_xyz[2]})",
                     os.path.join(args.output_dir, "4_translation_xyz.png"))
    print(f"[Q1b] Computed Translation along X, Y, Z: {t_xyz}")

    # 5. Combined Transformation (Rotation X + Y + Translation)
    R_combined = Ry @ Rx
    t_combined = [1.2, 0.8, 1.0]
    v_combined = apply_3d_transform(vertices, R=R_combined, t=t_combined)
    create_3d_figure(vertices, v_combined, edges,
                     "3D Transformation: Combined Rotation (X+Y) & Translation",
                     os.path.join(args.output_dir, "5_combined_transformation.png"))
    print("[Q1b] Computed Combined Rotation & Translation")

    # 6. Multi-panel figure showcasing all transformations together
    fig = plt.figure(figsize=(16, 10))
    demos = [
        ("Rotation around X (45°)", v_rot_x),
        ("Rotation around Y (45°)", v_rot_y),
        ("Rotation around Z (45°)", v_rot_z),
        (f"Translation ({t_xyz[0]}, {t_xyz[1]}, {t_xyz[2]})", v_trans),
        ("Combined Rotation + Translation", v_combined),
    ]

    limits = (-3, 3)
    for idx, (title, transformed_v) in enumerate(demos, 1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        plot_cube(ax, vertices, edges, color="royalblue", alpha=0.15, label="Original")
        plot_cube(ax, transformed_v, edges, color="crimson", alpha=0.4, label="Transformed")
        ax.plot3D([limits[0], limits[1]], [0, 0], [0, 0], "k--", linewidth=0.6, alpha=0.4)
        ax.plot3D([0, 0], [limits[0], limits[1]], [0, 0], "k--", linewidth=0.6, alpha=0.4)
        ax.plot3D([0, 0], [0, 0], [limits[0], limits[1]], "k--", linewidth=0.6, alpha=0.4)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        ax.view_init(elev=20, azim=45)
        if idx == 1:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, "all_3d_transformations_panel.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[Q1b] All 3D transformations generated and saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()


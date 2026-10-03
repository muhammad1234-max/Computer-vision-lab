"""
Task 8: Smart Security System – Boundary Detection & Intrusion Alarm
====================================================================
Goal: Detect boundaries of objects in a real-time video stream.
      Define a security zone; when an unauthorised object enters the zone,
      trigger an alarm (console message + visual alert).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# ── Security zone ─────────────────────────────────────────────────────────────

def define_security_zone(frame_shape, zone_rect=None):
    """
    Return the security zone as a polygon (numpy int32 array).

    Args:
        frame_shape : (h, w) or (h, w, c) of the video frame.
        zone_rect   : Optional (x, y, w, h) tuple for a rectangular zone.
                      Defaults to the centre third of the frame.

    Returns:
        zone_polygon : (4, 2) int32 array.
    """
    h, w = frame_shape[:2]
    if zone_rect is not None:
        x, y, zw, zh = zone_rect
    else:
        x  = w // 4
        y  = h // 4
        zw = w // 2
        zh = h // 2

    return np.array([[x, y], [x + zw, y], [x + zw, y + zh], [x, y + zh]], np.int32)


def point_in_zone(pt, zone_polygon):
    """Check whether a 2-D point is inside the zone polygon."""
    return cv2.pointPolygonTest(zone_polygon.reshape(-1, 1, 2), pt, False) >= 0


def bbox_overlaps_zone(bbox, zone_polygon):
    """Return True if any corner of bbox (x, y, w, h) is inside the zone."""
    x, y, bw, bh = bbox
    corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh),
               (x + bw // 2, y + bh // 2)]
    return any(point_in_zone(pt, zone_polygon) for pt in corners)


# ── Edge / contour detection ──────────────────────────────────────────────────

def detect_object_boundaries(frame, method="canny"):
    """
    Detect object boundaries in a frame.

    Args:
        frame  : BGR input frame.
        method : 'canny' or 'sobel'.

    Returns:
        edges   : Binary edge map.
        contours: Detected contours.
        bboxes  : Bounding rectangles of significant contours.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "canny":
        edges = cv2.Canny(blurred, 40, 120)
    else:
        sx    = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sy    = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag   = cv2.magnitude(sx, sy)
        edges = np.uint8(np.clip(mag / mag.max() * 255, 0, 255)) if mag.max() > 0 else np.zeros_like(gray)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Dilate to close small gaps
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    h, w = frame.shape[:2]
    min_area = h * w * 0.002
    significant = [c for c in contours if cv2.contourArea(c) > min_area]

    bboxes = [cv2.boundingRect(c) for c in significant]
    return edges, significant, bboxes


def detect_motion_objects(frame, bg_frame):
    """
    Detect moving/new objects using frame differencing (background subtraction).
    Returns diff image, contours, and bboxes of motion regions.
    """
    gray_f    = cv2.cvtColor(frame,    cv2.COLOR_BGR2GRAY)
    gray_bg   = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    diff      = cv2.absdiff(gray_bg, gray_f)
    blurred   = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 18, 255, cv2.THRESH_BINARY)
    kernel    = np.ones((5, 5), np.uint8)
    dilated   = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = h * w * 0.003
    significant = [c for c in contours if cv2.contourArea(c) > min_area]
    bboxes = [cv2.boundingRect(c) for c in significant]
    return thresh, significant, bboxes




# ── Frame annotation ──────────────────────────────────────────────────────────

def annotate_frame(frame, contours, bboxes, zone_polygon, alarm_active):
    """Draw contours, bounding boxes, zone, and alarm status on the frame."""
    annotated = frame.copy()

    # Draw zone
    zone_color = (0, 0, 220) if alarm_active else (0, 220, 0)
    cv2.polylines(annotated, [zone_polygon.reshape(-1, 1, 2)], True, zone_color, 3)

    # Transparent fill
    zone_fill = annotated.copy()
    fill_color = (0, 0, 100) if alarm_active else (0, 100, 0)
    cv2.fillPoly(zone_fill, [zone_polygon.reshape(-1, 1, 2)], fill_color)
    annotated  = cv2.addWeighted(annotated, 0.85, zone_fill, 0.15, 0)

    # Draw contours and bounding boxes
    for i, (cnt, bbox) in enumerate(zip(contours, bboxes)):
        x, y, bw, bh = bbox
        in_zone = bbox_overlaps_zone(bbox, zone_polygon)
        color   = (0, 0, 255) if in_zone else (255, 128, 0)
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        label = f"OBJ{i+1}" + (" ⚠" if in_zone else "")
        cv2.putText(annotated, label, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Zone label
    z_label = "SECURITY ZONE"
    tx, ty  = zone_polygon[0]
    cv2.putText(annotated, z_label, (tx + 5, ty + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

    # Alarm banner
    if alarm_active:
        banner_h = 45
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_h), (0, 0, 200), -1)
        cv2.putText(annotated, "⚠  ALARM: UNAUTHORISED OBJECT IN SECURITY ZONE  ⚠",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    else:
        cv2.putText(annotated, "Zone clear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    return annotated


# ── Synthetic video generator ─────────────────────────────────────────────────

def create_synthetic_security_frames(n_frames=40, h=480, w=640, seed=99):
    """
    Generate synthetic security-camera-like frames.
    An 'intruder' object moves into the security zone after frame 20.
    """
    rng    = np.random.default_rng(seed)
    frames = []

    # Background: corridor
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (50, 50, 55)
    # Floor line
    cv2.line(bg, (0, h * 2 // 3), (w, h * 2 // 3), (70, 70, 75), 2)
    # Walls texture
    for i in range(0, w, 60):
        cv2.line(bg, (i, 0), (i, h), (55, 55, 60), 1)

    for f in range(n_frames):
        frame = bg.copy()
        noise = rng.integers(0, 18, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Static object (always present, outside zone)
        cv2.rectangle(frame, (20, 200), (110, 340), (80, 100, 90), -1)
        cv2.rectangle(frame, (20, 200), (110, 340), (100, 130, 110), 2)

        # Intruder enters after frame 20
        progress = max(0, (f - 20) / 18)         # 0→1 over frames 20-38
        int_x    = int(w * 0.65 - progress * 250)
        int_y    = h // 4 + 20
        int_w    = 80
        int_h    = 120

        intruder_color = (40, 60, 160)
        cv2.rectangle(frame, (int_x, int_y), (int_x + int_w, int_y + int_h),
                      intruder_color, -1)
        # Person head
        cv2.circle(frame, (int_x + int_w // 2, int_y - 20), 18, intruder_color, -1)

        frames.append(frame)

    return frames


# ── Main processing loop ──────────────────────────────────────────────────────

def process_security_feed(frames, zone_polygon, method="canny"):
    """
    Process every frame using BOTH edge-based boundary detection and
    motion detection (frame differencing).  Alarm fires when a motion
    region overlaps the security zone.

    Returns:
        annotated_frames : List of annotated BGR frames.
        alarm_log        : List of (frame_idx, n_intruders) tuples where alarm fired.
    """
    annotated_frames = []
    alarm_log        = []

    # Use first frame as background reference
    bg_frame = frames[0].copy()

    for f_idx, frame in enumerate(frames):
        # 1. Edge-based boundaries (for display)
        edges, contours_e, bboxes_e = detect_object_boundaries(frame, method=method)

        # 2. Motion-based objects (for alarm)
        motion_map, contours_m, bboxes_m = detect_motion_objects(frame, bg_frame)

        # Combine contours for annotation (use motion bboxes for zone check)
        all_contours = contours_e + contours_m
        all_bboxes   = bboxes_e   + bboxes_m

        # Deduplicate overlapping bboxes (keep unique ones)
        unique_bboxes = list({b: None for b in all_bboxes}.keys())

        # Check intrusions using motion bboxes
        intruders    = [b for b in bboxes_m if bbox_overlaps_zone(b, zone_polygon)]
        alarm_active = len(intruders) > 0

        if alarm_active:
            alarm_log.append((f_idx + 1, len(intruders)))
            print(f"  !! ALARM | Frame {f_idx+1:3d} | "
                  f"{len(intruders)} object(s) in security zone")

        ann = annotate_frame(frame, contours_m, bboxes_m, zone_polygon, alarm_active)
        annotated_frames.append(ann)

    return annotated_frames, alarm_log


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 8: Smart Security System – Boundary Detection")
    print("=" * 60)

    n_frames      = 40
    frame_size    = (480, 640)
    frames        = create_synthetic_security_frames(n_frames=n_frames,
                                                     h=frame_size[0], w=frame_size[1])
    zone_polygon  = define_security_zone(frame_size, zone_rect=(160, 100, 320, 280))

    print(f"\nSecurity zone: {zone_polygon.tolist()}")
    print(f"Processing {n_frames} frames …")

    annotated, alarm_log = process_security_feed(frames, zone_polygon, method="canny")

    total_alarms = len(alarm_log)
    alarm_frames = [a[0] for a in alarm_log]
    print(f"\nAlarm triggered in {total_alarms} frame(s): {alarm_frames}")

    # ── Save annotated video ──────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    h, w   = annotated[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("outputs/task8_security_output.avi", fourcc, 8, (w, h))
    for f in annotated:
        writer.write(f)
    writer.release()
    print("Video saved → outputs/task8_security_output.avi")

    # ── Static visualisation ──────────────────────────────────────────────────
    # Pick representative frames
    sample_ids = [0, 10, 20, 25, 30, 38]
    sample_ids = [min(i, n_frames - 1) for i in sample_ids]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Task 8 – Smart Security System (Boundary Detection + Zone Intrusion)",
                 fontsize=13, fontweight="bold")

    for ax, idx in zip(axes.flat, sample_ids):
        ax.imshow(cv2.cvtColor(annotated[idx], cv2.COLOR_BGR2RGB))
        has_alarm = idx + 1 in alarm_frames
        ax.set_title(f"Frame {idx+1}  {'⚠ ALARM' if has_alarm else 'Clear'}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/task8_security_output.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Alarm timeline chart ──────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 3))
    all_frames   = list(range(1, n_frames + 1))
    alarm_binary = [1 if f in alarm_frames else 0 for f in all_frames]
    ax.fill_between(all_frames, alarm_binary, step="mid", alpha=0.6, color="red", label="Alarm")
    ax.set_xlabel("Frame"); ax.set_ylabel("Alarm Active")
    ax.set_title("Security Alarm Timeline"); ax.set_ylim(-0.1, 1.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/task8_alarm_timeline.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigures saved → outputs/task8_security_output.png, outputs/task8_alarm_timeline.png")


if __name__ == "__main__":
    main()

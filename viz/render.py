"""Visualization module for rendering hand skeletons, labels, and statistics."""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from common.schema import FrameHands, HandDetection

# Standard 21-point hand landmark skeleton connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Color palettes (BGR format)
# Left: Blue-Cyan, Right: Bright Amber-Orange
COLOR_LEFT = (235, 170, 0)       # Cyan-ish blue
COLOR_RIGHT = (0, 140, 255)      # Orange

COLOR_LEFT_DIM = (100, 70, 0)     # Dim Left
COLOR_RIGHT_DIM = (0, 60, 110)    # Dim Right


def render_frame_hands(
    frame: np.ndarray,
    frame_hands: FrameHands,
    running_det_share: float,
    running_dual_share: float,
    current_fps: float,
) -> np.ndarray:
    """Draw skeletons, labels, and HUD onto a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw each hand
    for hand in frame_hands.hands:
        is_pred = not hand.visible
        is_left = (hand.side.lower() == "left")

        if is_pred:
            color = COLOR_LEFT_DIM if is_left else COLOR_RIGHT_DIM
            joint_color = (80, 80, 80)
            thickness = 1
            radius = 2
        else:
            color = COLOR_LEFT if is_left else COLOR_RIGHT
            joint_color = (255, 255, 255)
            thickness = 2
            radius = 3

        kp2d = hand.kp2d

        # Draw skeleton connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(kp2d) and end_idx < len(kp2d):
                p1 = (int(round(kp2d[start_idx][0])), int(round(kp2d[start_idx][1])))
                p2 = (int(round(kp2d[end_idx][0])), int(round(kp2d[end_idx][1])))
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

        # Draw joints
        for pt in kp2d:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(out, (px, py), radius, joint_color, -1, cv2.LINE_AA)

        # Draw wrist label
        if len(kp2d) > 0:
            wrist = (int(round(kp2d[0][0])), int(round(kp2d[0][1])))
            side_str = "L" if is_left else "R"
            pred_tag = " (pred)" if is_pred else ""
            label_text = f"{side_str}: {hand.score:.2f}{pred_tag}"

            # Label background box
            label_pos = (wrist[0] - 10, max(20, wrist[1] + 18))
            cv2.putText(
                out,
                label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color if not is_pred else (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # Draw bounding box
        bx1, by1, bx2, by2 = [int(round(v)) for v in hand.bbox]
        box_color = color if not is_pred else (60, 60, 60)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), box_color, 1, cv2.LINE_AA)

    # HUD Overlay (top black banner for maximum readability)
    hud_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    hud_text = (
        f"Frame: {frame_hands.frame:04d} ({frame_hands.t:.2f}s) | "
        f"Hands: {len(frame_hands.hands)} | "
        f"Det Share: {running_det_share*100:.1f}% | "
        f"Dual: {running_dual_share*100:.1f}% | "
        f"{current_fps:.1f} FPS"
    )
    cv2.putText(
        out,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def ensure_h264_compatible(video_path: Path):
    """Re-encode video to standard H.264 with yuv420p via ffmpeg for universal playback."""
    import subprocess
    temp_path = video_path.with_name(f"{video_path.stem}_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "22",
        str(temp_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Replace original
        video_path.unlink()
        temp_path.rename(video_path)
    except Exception as e:
        print(f"Warning: ffmpeg re-encode failed: {e}")
        if temp_path.exists():
            temp_path.unlink()

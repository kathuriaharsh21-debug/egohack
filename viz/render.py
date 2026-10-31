"""Visualization module for rendering hand skeletons, labels, and statistics."""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from common.schema import FrameHands, HandDetection, FrameObjects, FrameContacts, ObjectDetection, HandContact

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
COLOR_LEFT = (235, 170, 0)       # Cyan-ish blue
COLOR_RIGHT = (0, 140, 255)      # Orange
COLOR_BYSTANDER = (140, 140, 140)# Gray for other people

COLOR_LEFT_DIM = (100, 70, 0)     # Dim Left
COLOR_RIGHT_DIM = (0, 60, 110)    # Dim Right


def render_frame_hands(
    frame: np.ndarray,
    frame_hands: FrameHands,
    running_det_share: float,
    running_dual_share: float,
    current_fps: float,
    frame_objects: Optional[FrameObjects] = None,
    frame_contacts: Optional[FrameContacts] = None,
) -> np.ndarray:
    """Draw skeletons, objects, contact states, and HUD onto a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Draw objects first (background layer)
    obj_by_id = {}
    if frame_objects and frame_objects.objects:
        for obj in frame_objects.objects:
            obj_by_id[obj.id] = obj
            bx1, by1, bx2, by2 = [int(round(v)) for v in obj.bbox]
            is_fixed = (obj.class_type == "fixed")
            obj_color = (255, 170, 0) if is_fixed else (0, 215, 255)  # Cyan for fixed, Gold for portable
            cv2.rectangle(out, (bx1, by1), (bx2, by2), obj_color, 1, cv2.LINE_AA)

            tag = f"{obj.label} #{obj.id}: {obj.score:.2f} [{obj.class_type}]"
            t_pos = (bx1, max(15, by1 - 4))
            cv2.putText(out, tag, t_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, tag, t_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.40, obj_color, 1, cv2.LINE_AA)

    # 2. Draw each hand
    contact_by_hand = {}
    if frame_contacts and frame_contacts.contacts:
        for c in frame_contacts.contacts:
            contact_by_hand[c.hand_id] = c

    for hand in frame_hands.hands:
        is_pred = not hand.visible
        is_left = (hand.side.lower() == "left")
        is_wearer = hand.is_wearer

        if not is_wearer:
            color = COLOR_BYSTANDER
            joint_color = (180, 180, 180)
            thickness = 1
            radius = 2
        elif is_pred:
            # Strictly GREY for predicted / invisible hands (never like real detections)
            color = (110, 110, 110)
            joint_color = (130, 130, 130)
            thickness = 1
            radius = 2
        else:
            color = COLOR_LEFT if is_left else COLOR_RIGHT
            joint_color = (255, 255, 255)
            thickness = 2
            radius = 3

        kp2d = hand.kp2d

        # Draw skeleton connections (dashed/dotted for predicted hands)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(kp2d) and end_idx < len(kp2d):
                p1 = (int(round(kp2d[start_idx][0])), int(round(kp2d[start_idx][1])))
                p2 = (int(round(kp2d[end_idx][0])), int(round(kp2d[end_idx][1])))
                if is_pred:
                    # Dotted effect: draw small circle at midpoint instead of solid line
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.line(out, p1, mid, color, 1, cv2.LINE_AA)
                else:
                    cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

        # Draw joints
        for pt in kp2d:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(out, (px, py), radius, joint_color, -1, cv2.LINE_AA)

        # Draw wrist label
        if len(kp2d) > 0:
            wrist = (int(round(kp2d[0][0])), int(round(kp2d[0][1])))
            side_str = "L" if is_left else "R"
            pred_tag = " (PRED-INVIS)" if is_pred else ""
            bystander_tag = " [other]" if not is_wearer else f" #{hand.id}"
            label_text = f"{side_str}{bystander_tag}: {hand.score:.2f}{pred_tag}"

            label_pos = (wrist[0] - 10, max(20, wrist[1] + 18))
            cv2.putText(out, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw bounding box
        bx1, by1, bx2, by2 = [int(round(v)) for v in hand.bbox]
        box_color = color if not is_pred else (90, 90, 90)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), box_color, 1, cv2.LINE_AA)

        # Draw contact link if active
        contact_info = contact_by_hand.get(hand.id)
        if contact_info and contact_info.state != "none" and not is_pred and is_wearer:
            # Find contacting object box
            obj_target = obj_by_id.get(contact_info.object_id)
            if obj_target:
                ox = int((obj_target.bbox[0] + obj_target.bbox[2]) / 2)
                oy = int((obj_target.bbox[1] + obj_target.bbox[3]) / 2)
                if len(kp2d) > 8:
                    hx = int(kp2d[8][0])  # Index fingertip
                    hy = int(kp2d[8][1])
                else:
                    hx = int((bx1 + bx2) / 2)
                    hy = int((by1 + by2) / 2)

                # Bright green contact line
                cv2.line(out, (hx, hy), (ox, oy), (0, 255, 0), 2, cv2.LINE_AA)
                c_tag = f"CONTACT: {contact_info.object_name} ({contact_info.state})"
                cv2.putText(out, c_tag, (hx, max(35, hy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # HUD Overlay (top black banner for maximum readability)
    hud_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    hud_text = (
        f"Frame: {frame_hands.frame:04d} ({frame_hands.t:.2f}s) | "
        f"Wearer Det: {running_det_share*100:.1f}% | "
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
        video_path.unlink()
        temp_path.rename(video_path)
    except Exception as e:
        print(f"Warning: ffmpeg re-encode failed: {e}")
        if temp_path.exists():
            temp_path.unlink()

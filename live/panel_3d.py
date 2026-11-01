"""3D skeleton panel using MediaPipe world landmarks.

Renders a rotated 3D hand skeleton as a small BGR image patch
suitable for inset display. Uses OpenCV only (no matplotlib) to
keep overhead minimal.

Label always shown: "MediaPipe 3D landmarks, not MANO"
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Hand connection pairs (same as MediaPipe's HAND_CONNECTIONS)
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),           # palm arch
]

_LABEL = "MediaPipe 3D landmarks, not MANO"


def render_3d(
    kp3d_list: List[List[List[float]]],
    sides: List[str],
    panel_w: int = 300,
    panel_h: int = 220,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Render 3D skeletons for one or two hands onto a BGR panel.

    Args:
        kp3d_list: list of kp3d arrays, each shape (21, 3) in metres
        sides: list of "left"/"right" strings matching kp3d_list
        panel_w, panel_h: output image size in pixels
        rotation_deg: slow rotation angle for animation (degrees)

    Returns:
        BGR numpy array of shape (panel_h, panel_w, 3)
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)  # dark background

    # Label at top
    cv2.putText(panel, _LABEL, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

    if not kp3d_list:
        cv2.putText(panel, "No hands", (panel_w // 2 - 35, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return panel

    colors = [(80, 200, 80), (200, 120, 80)]  # green for first, orange for second

    for idx, (kp3d, side) in enumerate(zip(kp3d_list, sides)):
        pts = np.array(kp3d, dtype=np.float32)  # (21, 3)
        if pts.shape != (21, 3):
            continue

        # Simple Y-axis rotation
        ang = math.radians(rotation_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
        pts = (rot @ pts.T).T

        # Normalise to panel
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # Centre and scale
        cx, cy = x.mean(), y.mean()
        scale = (min(panel_w, panel_h) * 0.38) / max(max(x) - min(x) + 1e-6,
                                                       max(y) - min(y) + 1e-6)

        # Map to pixel coords (flip Y so wrist at bottom)
        ox = panel_w // 2 + int((x - cx) * scale)
        oy = panel_h // 2 - int((y - cy) * scale)

        color = colors[idx % len(colors)]
        side_label = side.capitalize()

        # Draw bones
        for a, b in _CONNECTIONS:
            pa = (int(ox[a]), int(oy[a]))
            pb = (int(ox[b]), int(oy[b]))
            cv2.line(panel, pa, pb, color, 1, cv2.LINE_AA)

        # Draw joints (wrist larger)
        for j in range(21):
            radius = 4 if j == 0 else 2
            cv2.circle(panel, (int(ox[j]), int(oy[j])), radius, color, -1, cv2.LINE_AA)

        # Side label near wrist
        cv2.putText(panel, side_label, (int(ox[0]) + 3, int(oy[0]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return panel


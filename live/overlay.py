"""Drawing utilities for the live auto-annotator overlay.

All functions take a BGR numpy array and draw in-place.
No accuracy numbers are ever displayed.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from common.schema import FrameHands, HandDetection, ObjectDetection, FrameContacts
from live.event_log import ContactEvent

# Colour palette
COL_HAND_LEFT   = (80, 200, 80)    # green
COL_HAND_RIGHT  = (80, 120, 200)   # blue
COL_PRED        = (140, 140, 140)  # grey for memory/interpolated hands
COL_OBJECT      = (0, 200, 200)    # cyan
COL_FIXED_OBJ   = (0, 165, 255)   # orange for fixed surfaces
COL_CONTACT     = (0, 255, 128)    # bright green contact lines
COL_HUD         = (220, 220, 220)  # HUD text
COL_REVIEW      = (0, 100, 255)    # orange-red for low-confidence events
COL_EVENT_OK    = (150, 240, 150)  # light green for confirmed events

# MediaPipe hand connections
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ---------------------------------------------------------------------------
# Hand skeleton overlay
# ---------------------------------------------------------------------------

def draw_hands(frame: np.ndarray, frame_hands: FrameHands) -> None:
    """Draw skeletons for all tracked hands.

    Visible=True hands → solid colour.
    Visible=False (memory/predicted) → dashed grey with "PRED" label.
    """
    h, w = frame.shape[:2]
    for hand in frame_hands.hands:
        kp = hand.kp2d
        if not kp or len(kp) < 21:
            continue

        is_pred = not hand.visible
        color = COL_PRED if is_pred else (COL_HAND_LEFT if hand.side == "left" else COL_HAND_RIGHT)
        thickness = 1 if is_pred else 2

        # Bones
        for a, b in _CONNECTIONS:
            pa = (int(kp[a][0]), int(kp[a][1]))
            pb = (int(kp[b][0]), int(kp[b][1]))
            if is_pred:
                _draw_dashed_line(frame, pa, pb, color, thickness)
            else:
                cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

        # Joints
        for j, pt in enumerate(kp):
            r = 4 if j == 0 else 2
            cv2.circle(frame, (int(pt[0]), int(pt[1])), r, color, -1, cv2.LINE_AA)

        # Label: "L-1" / "R-2" + "PRED" badge
        wx, wy = int(kp[0][0]), int(kp[0][1])
        label = f"{'L' if hand.side == 'left' else 'R'}-{hand.id}"
        if is_pred:
            label += " PRED"
            cv2.putText(frame, label, (wx + 5, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, "predicted hand, not counted", (wx + 5, wy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, label, (wx + 5, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    steps = max(1, length // dash_len)
    for i in range(steps):
        if i % 2 == 0:
            t0 = i / steps
            t1 = (i + 0.5) / steps
            p0 = (int(x1 + t0 * (x2 - x1)), int(y1 + t0 * (y2 - y1)))
            p1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Object boxes
# ---------------------------------------------------------------------------

def draw_objects(
    frame: np.ndarray,
    objects: List[ObjectDetection],
    fixed_classes: Optional[set] = None,
) -> None:
    """Draw detected object bounding boxes."""
    fixed_classes = fixed_classes or set()
    for obj in objects:
        b = obj.bbox
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        is_fixed = obj.label.lower() in fixed_classes
        col = COL_FIXED_OBJ if is_fixed else COL_OBJECT
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        tag = f"{obj.label} {obj.score:.2f}"
        cv2.putText(frame, tag, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Contact lines
# ---------------------------------------------------------------------------

def draw_contacts(
    frame: np.ndarray,
    frame_contacts: FrameContacts,
    frame_hands: FrameHands,
    objects: List[ObjectDetection],
) -> None:
    """Draw lines from hand wrist to contacted object centre."""
    hand_map = {h.id: h for h in frame_hands.hands}
    obj_map = {o.obj_id: o for o in objects if hasattr(o, "obj_id")}

    for hc in frame_contacts.contacts:
        if hc.state == "none" or hc.object_id is None:
            continue
        hand = hand_map.get(hc.hand_id)
        if hand is None or not hand.visible or not hand.kp2d:
            continue

        wrist = (int(hand.kp2d[0][0]), int(hand.kp2d[0][1]))
        obj = obj_map.get(hc.object_id)
        if obj is None:
            # Try to draw state label only
            cv2.putText(frame, f"{hc.state.upper()} {hc.object_name}", wrist,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CONTACT, 1, cv2.LINE_AA)
            continue

        b = obj.bbox
        obj_cx = int((b[0] + b[2]) / 2)
        obj_cy = int((b[1] + b[3]) / 2)
        cv2.line(frame, wrist, (obj_cx, obj_cy), COL_CONTACT, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{hc.state.upper()}", (obj_cx, obj_cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CONTACT, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD (top-left corner)
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    fps: float,
    hand_ms: float,
    obj_ms: float,
    contact_ms: float,
    device: str,
    mode: str,
    frame_count: int,
    paused: bool = False,
    objects_on: bool = True,
    show_3d: bool = False,
    recording: bool = False,
    needs_review: bool = False,
) -> None:
    """Draw performance and mode HUD in the top-left corner.
    No accuracy numbers are ever shown.
    """
    lines = [
        f"FPS:{fps:4.1f}  Hand:{hand_ms:4.0f}ms  Obj:{obj_ms:4.0f}ms  Cntct:{contact_ms:3.0f}ms",
        f"Device:{device}  Mode:{mode}  Frame:{frame_count}",
    ]
    flags = []
    if paused:
        flags.append("PAUSED")
    if not objects_on:
        flags.append("OBJ-OFF")
    if show_3d:
        flags.append("3D-ON")
    if recording:
        flags.append("REC")
    if flags:
        lines.append("  ".join(flags))

    y0 = 18
    for i, line in enumerate(lines):
        y = y0 + i * 18
        cv2.putText(frame, line, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_HUD, 1, cv2.LINE_AA)

    if needs_review:
        badge_y = y0 + len(lines) * 18 + 4
        # Draw high-visibility NEEDS REVIEW badge
        cv2.rectangle(frame, (6, badge_y - 14), (160, badge_y + 4), (0, 0, 0), -1)
        cv2.rectangle(frame, (6, badge_y - 14), (160, badge_y + 4), COL_REVIEW, 2)
        cv2.putText(frame, "NEEDS REVIEW", (14, badge_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_REVIEW, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Event log panel (right side strip)
# ---------------------------------------------------------------------------

def draw_event_panel(
    frame: np.ndarray,
    events: List[ContactEvent],
    review_events: List[ContactEvent],
    panel_w: int = 220,
) -> None:
    """Draw the GRAB/RELEASE event log on the right side of the frame."""
    h, w = frame.shape[:2]
    # Dark background strip
    x0 = w - panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    title = "Event Log"
    cv2.putText(frame, title, (x0 + 6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    y = 34
    for ev in reversed(events[-10:]):
        dur = f"{(ev.t_end or 0) - ev.t_start:.1f}s"
        tag = f"{'L' if ev.side == 'left' else 'R'} {ev.state[:3].upper()} {ev.object_name[:10]} {dur}"
        col = COL_REVIEW if ev.needs_review else COL_EVENT_OK
        cv2.putText(frame, tag, (x0 + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        y += 16
        if y > h - 40:
            break

    # Review queue count
    if review_events:
        msg = f"Review queue: {len(review_events)}"
        cv2.putText(frame, msg, (x0 + 4, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_REVIEW, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Keyboard hint bar (bottom)
# ---------------------------------------------------------------------------

def draw_key_hints(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    hint = "Q:quit  S:switch  O:objects  3:3D  R:reset  P:pause  V:record"
    cv2.putText(frame, hint, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hint, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


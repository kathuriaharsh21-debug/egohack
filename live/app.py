"""Live Auto-Annotator — real-time hand-object contact demo.

Usage::

    python live/app.py --source webcam
    python live/app.py --source clip --clip mytronlabs-000003-013

Keys:
    Q   quit
    S   switch between webcam and clip
    O   toggle object detection
    3   toggle 3D panel
    R   reset event log
    P   pause / resume
    V   toggle video recording (saves to live/recordings/)

Architecture:
    - Capture thread  : newest-only frame buffer (never lags)
    - Hand thread     : MediaPipe VIDEO mode, increasing timestamps, num_hands=2
    - Object thread   : YOLO-World on GPU every 3rd frame, result reused between
    - Contact / draw  : runs in main loop using latest results from both threads
"""

import argparse
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── Path bootstrap so we can import from project root ──────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Suppress noisy MediaPipe / YOLO warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from common.schema import FrameHands, FrameContacts, HandDetection, ObjectDetection
from hands.detect import HandDetector
from objects.detector import ObjectDetector, get_scene_vocabulary_for_clip
from contacts.baseline import ContactEstimator
from live.capture import CaptureThread
from live.event_log import EventLog
from live import overlay as ov
from live.panel_3d import render_3d

# ── Generic webcam class list ───────────────────────────────────────────────
WEBCAM_CLASSES = [
    "tool", "bottle", "cup", "keyboard", "mouse", "pen", "book",
    "phone", "laptop", "box", "bag", "cloth", "cable", "part", "table"
]
WEBCAM_FIXED = {"table", "desk", "shelf"}

EVENT_PANEL_W = 220   # pixels reserved on the right for event log
PANEL_3D_W    = 300   # 3D panel width
PANEL_3D_H    = 220   # 3D panel height
FPS_SMOOTH    = 0.15  # EMA alpha for FPS display


# ── Config loader ───────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def resolve_clip_path(clip_name: str, config: dict) -> str:
    clips_dir = Path(config.get("data", {}).get("clips_480p_dir", "C:/egohack_data/clips_480p"))
    # Try exact match first, then glob
    for ext in (".mp4", ".avi", ".mkv"):
        p = clips_dir / (clip_name + ext)
        if p.exists():
            return str(p)
    matches = list(clips_dir.glob(f"*{clip_name}*"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Clip '{clip_name}' not found in {clips_dir}")


# ── Object detection thread ─────────────────────────────────────────────────

class ObjectThread(threading.Thread):
    """Runs YOLO-World on frames submitted via a queue (maxsize=1)."""

    def __init__(self, detector: ObjectDetector) -> None:
        super().__init__(daemon=True, name="ObjectThread")
        self.detector = detector
        self._in_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._last_result: List[ObjectDetection] = []
        self._last_ms: float = 0.0
        self._stop = threading.Event()

    def submit(self, frame_bgr: np.ndarray, wearer_bboxes: List[List[float]]) -> None:
        """Non-blocking submit: drop old frame if queue is full."""
        try:
            self._in_q.put_nowait((frame_bgr.copy(), wearer_bboxes))
        except queue.Full:
            pass

    def get_result(self) -> Tuple[List[ObjectDetection], float]:
        with self._result_lock:
            return list(self._last_result), self._last_ms

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame_bgr, wearer_bboxes = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                detections, _ = self.detector.detect_frame(frame_bgr, wearer_bboxes)
            except Exception as e:
                print(f"[ObjectThread] YOLO error: {e}")
                detections = []
            ms = (time.perf_counter() - t0) * 1000.0
            with self._result_lock:
                self._last_result = detections
                self._last_ms = ms


# ── Main application ─────────────────────────────────────────────────────────

def run(
    source: str = "webcam",
    clip_name: Optional[str] = None,
    config_path: str = "config.yaml",
) -> None:
    """Entry-point for the live demo."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    obj_cfg  = config.get("objects", {})
    ct_cfg   = obj_cfg.get("contact", {})

    # Determine scene vocabulary
    if source == "clip" and clip_name:
        classes, fixed_classes = get_scene_vocabulary_for_clip(clip_name, config)
        scene_label = clip_name
    else:
        classes = WEBCAM_CLASSES
        fixed_classes = WEBCAM_FIXED
        scene_label = "webcam"

    ignore_classes = set(c.lower() for c in obj_cfg.get("ignore_classes", []))

    # Clip path
    clip_path: Optional[str] = None
    if source == "clip" and clip_name:
        try:
            clip_path = resolve_clip_path(clip_name, config)
        except FileNotFoundError as e:
            print(f"[app] {e} — switching to webcam")
            source = "webcam"

    # ── Initialise modules ────────────────────────────────────────────────
    print(f"[app] Initialising HandDetector …")
    hand_model = data_cfg.get("weights_dir", "weights") + "/hand_landmarker.task"
    hand_det = HandDetector(
        model_path=hand_model,
        num_hands=2,
        min_detection_confidence=0.3,
    )

    print(f"[app] Initialising ObjectDetector (scene={scene_label}) …")
    try:
        obj_det = ObjectDetector(
            model_name=os.path.join(data_cfg.get("weights_dir", "weights"), "yolov8s-worldv2.pt"),
            classes=classes,
            conf_threshold=float(obj_cfg.get("confidence_threshold", 0.10)),
            crop_conf_threshold=float(obj_cfg.get("crop_confidence_threshold", 0.06)),
            fixed_classes=fixed_classes,
            ignore_classes=ignore_classes,
        )
        obj_ok = True
    except Exception as e:
        print(f"[app] WARNING: ObjectDetector failed ({e}). Running hands-only.")
        obj_det = None
        obj_ok = False

    import torch
    device_str = f"CUDA:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"

    print(f"[app] Initialising ContactEstimator …")
    contact_est = ContactEstimator(
        fixed_classes=fixed_classes,
        expansion_ratio=float(ct_cfg.get("expansion_ratio", 0.15)),
        min_fingertips=int(ct_cfg.get("min_fingertips", 2)),
        min_held_frames=int(ct_cfg.get("min_held_frames", 2)),
        min_hand_obj_iou=float(ct_cfg.get("min_hand_obj_iou", 0.08)),
        ignore_classes=ignore_classes,
    )

    event_log = EventLog(recordings_dir=str(_ROOT / "live" / "recordings"))

    # ── Capture thread ────────────────────────────────────────────────────
    cap = CaptureThread(source=source, clip_path=clip_path)
    cap.start()

    # Wait for first frame
    print("[app] Waiting for first frame …")
    for _ in range(50):
        frame, fidx, ts = cap.get_latest_frame()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("[app] ERROR: no frame received. Exiting.")
        cap.stop()
        hand_det.close()
        return

    h_frame, w_frame = frame.shape[:2]
    print(f"[app] Frame size: {w_frame}x{h_frame}  source={cap.source}")

    # ── Object thread ─────────────────────────────────────────────────────
    obj_thread: Optional[ObjectThread] = None
    if obj_ok and obj_det:
        obj_thread = ObjectThread(obj_det)
        obj_thread.start()

    # ── State ─────────────────────────────────────────────────────────────
    objects_on   = True
    show_3d      = False
    paused       = False
    recording    = False
    video_writer: Optional[cv2.VideoWriter] = None

    fps_smooth   = 30.0
    prev_t       = time.perf_counter()

    hand_ms      = 0.0
    obj_ms       = 0.0
    contact_ms   = 0.0

    last_objects: List[ObjectDetection] = []
    last_frame_contacts = FrameContacts(frame=0, t=0.0, contacts=[])
    last_frame_hands    = FrameHands(frame=0, t=0.0, hands=[])
    frame_counter       = 0
    rotation_deg        = 0.0

    cv2.namedWindow("Live Auto-Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Auto-Annotator", w_frame + EVENT_PANEL_W, h_frame)

    print("[app] Running. Press Q to quit.")

    while True:
        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Switch source
            cap.stop()
            source = "clip" if cap.source == "webcam" else "webcam"
            cap = CaptureThread(
                source=source,
                clip_path=clip_path,
                webcam_index=0,
            )
            cap.start()
        elif key == ord('o'):
            objects_on = not objects_on
        elif key == ord('3'):
            show_3d = not show_3d
        elif key == ord('r'):
            event_log.reset()
        elif key == ord('p'):
            paused = not paused
        elif key == ord('v'):
            if not recording:
                # Start recording
                import datetime as _dt
                rec_name = _dt.datetime.now().strftime("live_%Y%m%d_%H%M%S.avi")
                rec_path = str(_ROOT / "live" / "recordings" / rec_name)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out_w = w_frame + EVENT_PANEL_W
                video_writer = cv2.VideoWriter(rec_path, fourcc, 15.0, (out_w, h_frame))
                recording = True
                print(f"[app] Recording started: {rec_path}")
                print("[app] ⚠ SHARING WARNING: recording contains real-world video. Keep private per data rules.")
            else:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                recording = False
                print("[app] Recording stopped.")

        if paused:
            # Still show last frame
            cv2.imshow("Live Auto-Annotator", _last_display if '_last_display' in dir() else frame)
            continue

        # ── Grab newest frame ──────────────────────────────────────────────
        frame, fidx, ts = cap.get_latest_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1

        # ── Hand detection (every frame) ────────────────────────────────
        t0 = time.perf_counter()
        try:
            last_frame_hands = hand_det.process_frame(
                frame_bgr=frame,
                frame_idx=fidx,
                timestamp_sec=ts,
                width=w_frame,
                height=h_frame,
            )
        except Exception as e:
            print(f"[app] Hand detect error: {e}")
        hand_ms = (time.perf_counter() - t0) * 1000.0

        # Wearer hand bboxes for object second-pass
        wearer_bboxes = [
            h.bbox for h in last_frame_hands.hands
            if h.visible and h.is_wearer and h.bbox
        ]

        # ── Object detection (every 3rd frame via thread) ───────────────
        if objects_on and obj_thread and frame_counter % 3 == 0:
            obj_thread.submit(frame, wearer_bboxes)

        if objects_on and obj_thread:
            last_objects, obj_ms = obj_thread.get_result()
        else:
            last_objects = []
            obj_ms = 0.0

        # ── Contact estimation ──────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            wearer_hands = [h for h in last_frame_hands.hands if h.visible and h.is_wearer]
            last_frame_contacts = contact_est.process_frame(
                frame_idx=fidx,
                timestamp_sec=ts,
                wearer_hands=wearer_hands,
                detected_objects=last_objects,
            )
            event_log.update(last_frame_contacts)
        except Exception as e:
            print(f"[app] Contact error: {e}")
        contact_ms = (time.perf_counter() - t0) * 1000.0

        # ── FPS ─────────────────────────────────────────────────────────
        now = time.perf_counter()
        instant_fps = 1.0 / max(1e-6, now - prev_t)
        fps_smooth = (1 - FPS_SMOOTH) * fps_smooth + FPS_SMOOTH * instant_fps
        prev_t = now

        # ── Draw ─────────────────────────────────────────────────────────
        display = frame.copy()

        ov.draw_hands(display, last_frame_hands)
        if objects_on:
            ov.draw_objects(display, last_objects, fixed_classes=fixed_classes)
        ov.draw_contacts(display, last_frame_contacts, last_frame_hands, last_objects)
        ov.draw_hud(
            display, fps=fps_smooth,
            hand_ms=hand_ms, obj_ms=obj_ms, contact_ms=contact_ms,
            device=device_str, mode=cap.source, frame_count=frame_counter,
            paused=paused, objects_on=objects_on,
            show_3d=show_3d, recording=recording,
        )
        ov.draw_key_hints(display)

        # ── Event panel (right strip) ────────────────────────────────────
        # Expand canvas
        canvas = np.zeros((h_frame, w_frame + EVENT_PANEL_W, 3), dtype=np.uint8)
        canvas[:, :w_frame] = display
        panel_slice = canvas[:, w_frame:]
        ov.draw_event_panel(
            canvas,
            event_log.get_display_events(),
            event_log.get_review_events(),
            panel_w=EVENT_PANEL_W,
        )

        # ── 3D panel inset ────────────────────────────────────────────────
        if show_3d:
            rotation_deg = (rotation_deg + 0.5) % 360.0
            vis_hands = [h for h in last_frame_hands.hands if h.visible and h.kp3d]
            kp3d_list = [h.kp3d for h in vis_hands]
            sides     = [h.side for h in vis_hands]
            panel_img = render_3d(kp3d_list, sides,
                                  panel_w=PANEL_3D_W, panel_h=PANEL_3D_H,
                                  rotation_deg=rotation_deg)
            # Inset bottom-left
            y1 = h_frame - PANEL_3D_H
            x2 = PANEL_3D_W
            if y1 >= 0 and x2 <= w_frame:
                alpha = 0.85
                roi = canvas[y1:h_frame, 0:x2]
                cv2.addWeighted(panel_img, alpha, roi, 1 - alpha, 0, roi)
                canvas[y1:h_frame, 0:x2] = roi

        _last_display = canvas
        cv2.imshow("Live Auto-Annotator", canvas)

        if recording and video_writer:
            try:
                video_writer.write(canvas)
            except Exception:
                pass

    # ── Cleanup ──────────────────────────────────────────────────────────
    print("[app] Shutting down …")
    cap.stop()
    if obj_thread:
        obj_thread.stop()
    if video_writer:
        video_writer.release()
    hand_det.close()
    cv2.destroyAllWindows()
    print(f"[app] Done. Event CSV: {event_log.csv_path}")


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Auto-Annotator")
    parser.add_argument("--source", choices=["webcam", "clip"], default="webcam",
                        help="Input source: 'webcam' or 'clip'")
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip stem name (e.g. mytronlabs-000003-013)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    if args.source == "clip" and not args.clip:
        parser.error("--clip is required when --source=clip")

    run(source=args.source, clip_name=args.clip, config_path=args.config)

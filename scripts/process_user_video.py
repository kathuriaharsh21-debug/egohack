"""Process a user-provided video using the high-accuracy hand & object contact pipeline.

Features:
- Hand tracking (MediaPipe Tasks HandLandmarker + Hungarian tracking + 1-Euro smoothing)
- Open-vocabulary object detection (YOLO-World v2 on CUDA)
- Strict fingertip-touch gating (object boxed only when touched)
- Contact state estimation ('portable object' vs 'fixed surface')
- Renders full overlay video to user_processing/overlay/
- Saves keyframe snapshot images of every touch event
- Generates structured JSON with touch start, end, duration, and object names to user_processing/output/

Usage:
    python scripts/process_user_video.py
    python scripts/process_user_video.py --video user_processing/original/my_test.mp4
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.schema import FrameHands, FrameContacts, HandDetection, ObjectDetection, HandContact
from hands.detect import HandDetector
from objects.detector import ObjectDetector
from contacts.baseline import ContactEstimator
from live import overlay as ov

# High-accuracy prompt vocabulary for tools, devices, workpieces & surfaces
TARGET_CLASSES = [
    "phone", "cell phone", "pen", "tool", "cup", "bottle",
    "mouse", "keyboard", "book", "box", "held object", "part",
    "component", "workpiece", "table", "desk"
]
FIXED_CLASSES = {"table", "desk", "surface"}


def process_video(
    video_path: Path,
    out_dir_overlay: Path,
    out_dir_json: Path,
    conf_thresh: float = 0.08,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    stem = video_path.stem
    out_dir_overlay.mkdir(parents=True, exist_ok=True)
    out_dir_json.mkdir(parents=True, exist_ok=True)

    out_video_path = out_dir_overlay / f"{stem}_overlay.mp4"
    out_json_path = out_dir_json / f"{stem}_touch_events.json"

    print("=======================================================")
    print(f" Processing User Video: {video_path.name}")
    print(f" Source Path: {video_path.resolve()}")
    print("=======================================================")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_total = total_frames / max(1.0, fps)

    print(f" Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames} ({duration_total:.2f}s)")

    # Initialize models
    weights_dir = REPO_ROOT / "weights"
    hand_task = weights_dir / "hand_landmarker.task"
    if not hand_task.exists():
        raise FileNotFoundError(f"Hand landmarker task not found at {hand_task}")

    print(" Loading MediaPipe Hand Landmarker …")
    hand_det = HandDetector(
        model_path=str(hand_task),
        num_hands=2,
        min_detection_confidence=0.3,
    )

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f" Loading YOLO-World v2 on {device} …")
    yolo_model_path = weights_dir / "yolov8s-worldv2.pt"
    obj_det = ObjectDetector(
        model_name=str(yolo_model_path) if yolo_model_path.exists() else "yolov8s-worldv2.pt",
        classes=TARGET_CLASSES,
        conf_threshold=conf_thresh,
        crop_conf_threshold=0.04,
        fixed_classes=FIXED_CLASSES,
        ignore_classes=set(),  # Do not suppress phones or tools
    )

    contact_est = ContactEstimator(
        fixed_classes=FIXED_CLASSES,
        expansion_ratio=0.22,
        min_fingertips=1,
        min_held_frames=2,
        min_hand_obj_iou=0.05,
        ignore_classes=set(),
    )

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Event tracking
    active_touch_events: Dict[int, Dict[str, Any]] = {}  # hand_id -> current touch event dict
    completed_touch_events: List[Dict[str, Any]] = []

    frame_idx = 0
    t0_all = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / max(1.0, fps)

        # 1. Hand detection
        try:
            if frame_idx % 2 == 0:
                frame_hands = hand_det.process_frame(
                    frame_bgr=frame,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    width=width,
                    height=height,
                )
            else:
                frame_hands = hand_det.process_skipped_frame(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                )
        except Exception as e:
            frame_hands = FrameHands(frame=frame_idx, t=timestamp_sec, hands=[])

        # Treat all hands as wearer for user videos
        for h in frame_hands.hands:
            h.is_wearer = True

        wearer_bboxes = [h.bbox for h in frame_hands.hands if h.visible and h.bbox]

        # 2. Object detection (run on every frame or every 2nd frame)
        try:
            detected_objects, _ = obj_det.detect_frame(frame, wearer_bboxes)
        except Exception:
            detected_objects = []

        # 3. Contact estimation
        try:
            frame_contacts = contact_est.process_frame(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                wearer_hands=frame_hands.hands,
                detected_objects=detected_objects,
            )
        except Exception:
            frame_contacts = FrameContacts(frame=frame_idx, t=timestamp_sec, contacts=[])

        # 4. Touch Event State Machine (GRAB -> HOLD -> RELEASE)
        current_contact_hand_ids = set()
        for c in frame_contacts.contacts:
            current_contact_hand_ids.add(c.hand_id)
            if c.state != "none" and c.object_name != "none":
                if c.hand_id not in active_touch_events:
                    # New Touch Started
                    ev_id = len(completed_touch_events) + len(active_touch_events) + 1
                    active_touch_events[c.hand_id] = {
                        "event_id": ev_id,
                        "object_name": c.object_name,
                        "state": c.state,
                        "hand_side": c.side,
                        "hand_id": c.hand_id,
                        "t_start": round(timestamp_sec, 3),
                        "t_end": None,
                        "duration_sec": None,
                        "start_frame": frame_idx,
                        "end_frame": None,
                        "confidence_scores": [float(c.confidence)],
                        "keyframe_image": None,
                        "peak_frame": frame.copy(),
                    }
                else:
                    # Touch continues
                    active_touch_events[c.hand_id]["confidence_scores"].append(float(c.confidence))
            else:
                # Hand in frame but released contact
                if c.hand_id in active_touch_events:
                    ev = active_touch_events.pop(c.hand_id)
                    ev["t_end"] = round(timestamp_sec, 3)
                    ev["end_frame"] = frame_idx
                    ev["duration_sec"] = round(ev["t_end"] - ev["t_start"], 3)
                    ev["confidence"] = round(float(sum(ev["confidence_scores"]) / max(1, len(ev["confidence_scores"]))), 3)
                    del ev["confidence_scores"]

                    # Save snapshot image at touch release
                    img_name = f"{stem}_touch_{ev['event_id']}_{ev['object_name']}_{ev['t_start']:.1f}s.jpg"
                    img_path = out_dir_overlay / img_name
                    cv2.imwrite(str(img_path), ev["peak_frame"])
                    ev["keyframe_image"] = str(img_path.name)
                    del ev["peak_frame"]

                    completed_touch_events.append(ev)
                    print(f" [TOUCH EVENT {ev['event_id']}] {ev['object_name'].upper()} ({ev['state']}) | {ev['t_start']}s -> {ev['t_end']}s ({ev['duration_sec']}s)")

        # Close any hand that left the frame
        for hid in list(active_touch_events):
            if hid not in current_contact_hand_ids:
                ev = active_touch_events.pop(hid)
                ev["t_end"] = round(timestamp_sec, 3)
                ev["end_frame"] = frame_idx
                ev["duration_sec"] = round(ev["t_end"] - ev["t_start"], 3)
                ev["confidence"] = round(float(sum(ev["confidence_scores"]) / max(1, len(ev["confidence_scores"]))), 3)
                del ev["confidence_scores"]

                img_name = f"{stem}_touch_{ev['event_id']}_{ev['object_name']}_{ev['t_start']:.1f}s.jpg"
                img_path = out_dir_overlay / img_name
                cv2.imwrite(str(img_path), ev["peak_frame"])
                ev["keyframe_image"] = str(img_path.name)
                del ev["peak_frame"]

                completed_touch_events.append(ev)

        # 5. Render Frame Overlay
        out_frame = frame.copy()
        ov.draw_hands(out_frame, frame_hands)
        ov.draw_objects(out_frame, detected_objects, fixed_classes=FIXED_CLASSES, frame_hands=frame_hands, contact_tolerance_px=10)
        ov.draw_contacts(out_frame, frame_contacts, frame_hands, detected_objects)

        # Draw HUD info
        active_str = " | ".join([f"{ev['object_name'].upper()} ({ev['state']})" for ev in active_touch_events.values()])
        cv2.putText(out_frame, f"Time: {timestamp_sec:.2f}s / {duration_total:.1f}s", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if active_str:
            cv2.putText(out_frame, f"TOUCHING: {active_str}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(out_frame)
        frame_idx += 1

        if frame_idx % 60 == 0 or frame_idx == total_frames:
            pct = (frame_idx / max(1, total_frames)) * 100
            sys.stdout.write(f"\r Processing: {frame_idx}/{total_frames} frames ({pct:.1f}%)")
            sys.stdout.flush()

    # Finalize any ongoing touch at end of video
    final_ts = round(total_frames / max(1.0, fps), 3)
    for hid, ev in active_touch_events.items():
        ev["t_end"] = final_ts
        ev["end_frame"] = total_frames
        ev["duration_sec"] = round(ev["t_end"] - ev["t_start"], 3)
        ev["confidence"] = round(float(sum(ev["confidence_scores"]) / max(1, len(ev["confidence_scores"]))), 3)
        del ev["confidence_scores"]
        img_name = f"{stem}_touch_{ev['event_id']}_{ev['object_name']}_{ev['t_start']:.1f}s.jpg"
        img_path = out_dir_overlay / img_name
        cv2.imwrite(str(img_path), ev["peak_frame"])
        ev["keyframe_image"] = str(img_path.name)
        del ev["peak_frame"]
        completed_touch_events.append(ev)

    cap.release()
    writer.release()
    hand_det.close()

    elap_total = time.perf_counter() - t0_all
    proc_fps = total_frames / max(1e-3, elap_total)

    # 6. Save JSON Output
    output_summary = {
        "video_file": video_path.name,
        "total_duration_sec": round(duration_total, 3),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "processing_time_sec": round(elap_total, 2),
        "processing_fps": round(proc_fps, 2),
        "overlay_video": str(out_video_path.name),
        "touch_events_count": len(completed_touch_events),
        "touch_events": completed_touch_events,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n\n=======================================================")
    print(" Video Processing Completed Successfully!")
    print(f" Output Overlay Video: {out_video_path.resolve()}")
    print(f" Output JSON Events:   {out_json_path.resolve()}")
    print(f" Touch Events Count:   {len(completed_touch_events)}")
    print(f" Total Processing Time: {elap_total:.1f}s ({proc_fps:.1f} FPS)")
    print("=======================================================")

    print("\n--- TOUCH EVENTS JSON SUMMARY ---")
    print(json.dumps(output_summary, indent=2))
    print("---------------------------------\n")

    return out_video_path, out_json_path


def main():
    parser = argparse.ArgumentParser(description="Process user video with hand & object contact pipeline")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input mp4/avi video file (default: newest in user_processing/original/)")
    parser.add_argument("--overlay-dir", type=str, default="user_processing/overlay",
                        help="Directory to save overlay video and snapshot images")
    parser.add_argument("--json-dir", type=str, default="user_processing/output",
                        help="Directory to save touch timestamps JSON")
    parser.add_argument("--conf", type=float, default=0.08,
                        help="YOLO-World confidence threshold (default 0.08)")
    args = parser.parse_args()

    overlay_dir = Path(args.overlay_dir)
    json_dir = Path(args.json_dir)
    orig_dir = Path("user_processing/original")
    orig_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    if args.video:
        video_path = Path(args.video)
    else:
        # Auto-detect latest video in user_processing/original/
        videos = sorted(list(orig_dir.glob("*.mp4")) + list(orig_dir.glob("*.avi")) + list(orig_dir.glob("*.mov")),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            video_path = videos[0]
            print(f"[AUTO-DETECT] Found video in {orig_dir}: {video_path.name}")
        else:
            print(f"No video specified and no video found in '{orig_dir}'.")
            print(f"Please copy or upload your video into '{orig_dir.resolve()}' and run again:")
            print(f"    python scripts/process_user_video.py --video user_processing/original/your_video.mp4")
            sys.exit(1)

    process_video(video_path, overlay_dir, json_dir, conf_thresh=args.conf)


if __name__ == "__main__":
    main()

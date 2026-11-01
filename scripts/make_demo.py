"""Generate a 2-minute demonstration video with timeline strip and overlay.

Outputs to D:/egohack_data/demo/egohack_demo_2min.mp4
"""

import sys
import os
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import FrameHands, FrameObjects, FrameContacts, HandDetection, ObjectDetection, HandContact
from viz.render import render_frame_hands, ensure_h264_compatible

def draw_timeline_strip(frame: np.ndarray, current_sec: float, total_sec: float, history: list) -> np.ndarray:
    """Draw a 24px timeline interaction strip along the bottom of the frame."""
    fh, fw = frame.shape[:2]
    strip_h = 24
    y1 = fh - strip_h
    # Dark background
    cv2.rectangle(frame, (0, y1), (fw, fh), (30, 30, 30), -1)

    # State colors (BGR)
    color_map = {
        "none": (60, 60, 60),        # dark gray
        "portable": (0, 200, 255),    # bright yellow/gold
        "fixed": (255, 150, 0)        # cyan-blue
    }

    # Draw historical segments
    for seg in history:
        t0, t1, st = seg["t0"], seg["t1"], seg["state"]
        x0 = int((t0 / total_sec) * fw)
        x1 = int((t1 / total_sec) * fw)
        col = color_map.get(st, (60, 60, 60))
        cv2.rectangle(frame, (x0, y1 + 2), (max(x0 + 1, x1), fh - 2), col, -1)

    # Current time scrubber marker
    cur_x = int((current_sec / total_sec) * fw)
    cv2.line(frame, (cur_x, y1), (cur_x, fh), (0, 0, 255), 2)

    # Label
    cv2.putText(frame, f"Timeline: {current_sec:.1f}s / {total_sec:.0f}s", (10, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
    return frame

def make_demo(
    clip_name: str = "mytronlabs-000060-006",
    duration_sec: float = 120.0,
    out_dir: Path = Path("D:/egohack_data/demo"),
    clips_dir: Path = Path("data/clips"),
    pipeline_out: Path = Path("out")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "egohack_demo_2min.mp4"

    clip_video = clips_dir / f"{clip_name}.mp4"
    if not clip_video.exists():
        print(f"Error: {clip_video} does not exist.")
        return

    # Load pre-computed outputs
    hands_file = pipeline_out / clip_name / "hands.jsonl"
    objs_file = pipeline_out / clip_name / "objects.jsonl"
    contacts_file = pipeline_out / clip_name / "contacts_frames.jsonl"

    hands_by_frame = {}
    if hands_file.exists():
        with open(hands_file, "r", encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                hands_by_frame[d["frame"]] = d

    objs_by_frame = {}
    if objs_file.exists():
        with open(objs_file, "r", encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                objs_by_frame[d["frame"]] = d

    contacts_by_frame = {}
    if contacts_file.exists():
        with open(contacts_file, "r", encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                contacts_by_frame[d["frame"]] = d

    # Pre-build timeline history segments
    timeline_history = []
    max_frame = int(duration_sec * 30)

    cur_state = "none"
    seg_start = 0.0
    for f_idx in range(max_frame):
        t_sec = f_idx / 30.0
        cf_d = contacts_by_frame.get(f_idx, {})
        # Pick dominant wearer state
        st = "none"
        for c in cf_d.get("contacts", []):
            if c.get("state") in ["portable", "fixed"]:
                st = c["state"]
                break
        if st != cur_state:
            timeline_history.append({"t0": seg_start, "t1": t_sec, "state": cur_state})
            cur_state = st
            seg_start = t_sec
    timeline_history.append({"t0": seg_start, "t1": duration_sec, "state": cur_state})

    cap = cv2.VideoCapture(str(clip_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    print(f"Rendering demo video for {clip_name}: {total_frames} frames ({duration_sec}s)...")
    t0 = time.perf_counter()

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = f_idx / fps

        # Build schema objects
        h_data = hands_by_frame.get(f_idx, {})
        fh = FrameHands(
            frame=f_idx,
            t=t_sec,
            hands=[HandDetection(**h) for h in h_data.get("hands", [])]
        )

        o_data = objs_by_frame.get(f_idx, {})
        fo = FrameObjects(
            frame=f_idx,
            t=t_sec,
            objects=[ObjectDetection(**o) for o in o_data.get("objects", [])]
        )

        c_data = contacts_by_frame.get(f_idx, {})
        fc = FrameContacts(
            frame=f_idx,
            t=t_sec,
            contacts=[HandContact(**c) for c in c_data.get("contacts", [])]
        )

        # Overlay
        out_frame = render_frame_hands(
            frame=frame,
            frame_hands=fh,
            running_det_share=0.45,
            running_dual_share=0.15,
            current_fps=fps,
            frame_objects=fo,
            frame_contacts=fc
        )

        # Timeline strip
        out_frame = draw_timeline_strip(out_frame, t_sec, duration_sec, timeline_history)

        writer.write(out_frame)
        if (f_idx + 1) % 600 == 0:
            print(f"  Frame {f_idx+1}/{total_frames} ({((f_idx+1)/total_frames)*100:.1f}%)")

    cap.release()
    writer.release()
    elap = time.perf_counter() - t0
    ensure_h264_compatible(out_mp4)
    print(f"Successfully generated {out_mp4} ({total_frames} frames, {elap:.1f}s, size: {out_mp4.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    make_demo()


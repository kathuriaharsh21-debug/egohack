"""Batch pipeline runner: Decoupled Stage A (Hands) and Stage B (Objects).

Stage A:
- MediaPipe VIDEO mode on CPU, step=2, overlay off.
- Resumable: skips if hands.jsonl already exists and has 5400 valid frames.
- Writes hands.jsonl per clip.

Stage B:
- YOLO-World on GPU (cuda:0).
- Uses per-scene vocabulary and ignore list from config.yaml.
- Order: 4 dev clips first, then remaining clips.
- Writes objects.jsonl per clip.

Stage C (Contact) remains a separate decoupled step that can be rerun at any time on CPU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import datetime
import json
import cv2
import yaml
import torch

from common.io import frame_generator
from common.schema import FrameHands, FrameObjects
from common.validate import validate_hands_jsonl, validate_objects_jsonl
from hands.detect import HandDetector
from objects.detector import ObjectDetector, get_scene_vocabulary_for_clip
from run_pipeline import load_config


def get_all_clip_paths(clips_dir: Path):
    return sorted(list(clips_dir.glob("*.mp4")))


def run_stage_a_clip(clip_path: Path, out_dir: Path, step: int = 2) -> dict:
    clip_stem = clip_path.stem
    clip_out = out_dir / clip_stem
    clip_out.mkdir(parents=True, exist_ok=True)
    hands_jsonl_path = clip_out / "hands.jsonl"

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Resumable check: if hands.jsonl exists and is valid with total_frames
    if hands_jsonl_path.exists():
        valid, errs = validate_hands_jsonl(hands_jsonl_path)
        if valid:
            with open(hands_jsonl_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count >= total_frames:
                return {"status": "skipped", "frames": line_count, "time": 0.0, "fps": 0.0}

    detector = HandDetector(
        model_path="weights/hand_landmarker.task",
        num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        filter_cutoff=1.0,
        filter_beta=0.01,
    )

    hands_lines = []
    t0 = time.perf_counter()
    for frame_idx, t_sec, frame in frame_generator(clip_path, step=1):
        if frame_idx % step == 0:
            fh = detector.process_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                width=width,
                height=height,
            )
        else:
            fh = detector.process_skipped_frame(frame_idx=frame_idx, timestamp_sec=t_sec)
        hands_lines.append(fh.to_dict())

    detector.close()
    elap = time.perf_counter() - t0

    tmp_path = clip_out / "hands.jsonl.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in hands_lines:
            f.write(json.dumps(item) + "\n")
    if hands_jsonl_path.exists():
        hands_jsonl_path.unlink()
    tmp_path.rename(hands_jsonl_path)

    fps_run = len(hands_lines) / max(1e-4, elap)
    return {"status": "done", "frames": len(hands_lines), "time": elap, "fps": fps_run}


def run_stage_b_clip(clip_path: Path, out_dir: Path, config: dict, step: int = 2, force: bool = False) -> dict:
    clip_stem = clip_path.stem
    clip_out = out_dir / clip_stem
    clip_out.mkdir(parents=True, exist_ok=True)
    objects_jsonl_path = clip_out / "objects.jsonl"
    hands_jsonl_path = clip_out / "hands.jsonl"

    if not hands_jsonl_path.exists():
        raise FileNotFoundError(f"Cannot run Stage B for {clip_stem}: hands.jsonl does not exist")

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not force and objects_jsonl_path.exists():
        valid, errs = validate_objects_jsonl(objects_jsonl_path)
        if valid:
            with open(objects_jsonl_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count >= total_frames:
                return {"status": "skipped", "frames": line_count, "time": 0.0, "fps": 0.0}

    # Load wearer bboxes from hands.jsonl
    wearer_bboxes_by_frame = {}
    with open(hands_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            fidx = data.get("frame", 0)
            bboxes = [
                h["bbox"] for h in data.get("hands", [])
                if h.get("visible", False) and h.get("is_wearer", True) and h.get("bbox")
            ]
            wearer_bboxes_by_frame[fidx] = bboxes

    obj_cfg = config.get("objects", {})
    scene_classes, scene_fixed = get_scene_vocabulary_for_clip(clip_stem, config)
    ignore_classes = set(c.lower() for c in obj_cfg.get("ignore_classes", []))

    obj_detector = ObjectDetector(
        model_name=obj_cfg.get("model", "yolov8s-worldv2.pt"),
        classes=scene_classes,
        conf_threshold=float(obj_cfg.get("confidence_threshold", 0.10)),
        crop_conf_threshold=float(obj_cfg.get("crop_confidence_threshold", 0.06)),
        hand_iou_threshold=float(obj_cfg.get("hand_object_iou_threshold", 0.6)),
        fixed_classes=scene_fixed,
        ignore_classes=ignore_classes,
    )

    objects_lines = []
    last_objs = []
    t0 = time.perf_counter()

    for frame_idx, t_sec, frame in frame_generator(clip_path, step=1):
        if frame_idx % step == 0:
            w_boxes = wearer_bboxes_by_frame.get(frame_idx, [])
            objs, _ = obj_detector.detect_frame(frame, w_boxes)
            last_objs = objs
        else:
            objs = last_objs

        fo = FrameObjects(frame=frame_idx, t=t_sec, objects=objs)
        objects_lines.append(fo.to_dict())

    elap = time.perf_counter() - t0

    tmp_path = clip_out / "objects.jsonl.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in objects_lines:
            f.write(json.dumps(item) + "\n")
    if objects_jsonl_path.exists():
        objects_jsonl_path.unlink()
    tmp_path.rename(objects_jsonl_path)

    fps_run = len(objects_lines) / max(1e-4, elap)
    return {"status": "done", "frames": len(objects_lines), "time": elap, "fps": fps_run}


def main():
    config = load_config()
    clips_dir = Path(config.get("data", {}).get("clips_480p_dir", "C:/egohack_data/clips_480p"))
    out_dir = Path(config.get("data", {}).get("out_dir", "D:/egohack_data/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = Path("PROGRESS.md")

    clips = get_all_clip_paths(clips_dir)
    print(f"Found {len(clips)} clips in {clips_dir}")

    # =========================================================================
    # STAGE A: Hands only (CPU, step=2, resumable)
    # =========================================================================
    start_a_dt = datetime.datetime.now()
    print(f"\n>>> Starting STAGE A (Hands Only) at {start_a_dt.strftime('%H:%M:%S')} IST")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"\n### Batch Stage A (Hands) Started: {start_a_dt.strftime('%Y-%m-%d %H:%M:%S')} IST\n")

    t_start_a = time.perf_counter()
    stage_a_done_count = 0
    for idx, c in enumerate(clips, 1):
        stem = c.stem
        res = run_stage_a_clip(c, out_dir, step=2)
        if res["status"] == "skipped":
            print(f"[{idx}/40] Stage A SKIPPED (already complete): {stem}")
        else:
            stage_a_done_count += 1
            line = f"- [Stage A {idx:02d}/40] `{stem}`: {res['time']:.1f}s ({res['time']/60:.2f}m) | {res['fps']:.1f} FPS | {res['frames']} frames\n"
            print(f"[{idx}/40] Stage A DONE: {stem} in {res['time']:.1f}s ({res['fps']:.1f} FPS)")
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(line)

    elap_a = time.perf_counter() - t_start_a
    print(f">>> STAGE A COMPLETED in {elap_a/60:.2f} min ({stage_a_done_count} clips processed, {len(clips)-stage_a_done_count} skipped)")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"- Stage A Finished: {datetime.datetime.now().strftime('%H:%M:%S')} IST (processed {stage_a_done_count} clips in {elap_a/60:.2f} min)\n")

    # =========================================================================
    # STAGE B: Objects (GPU, order: 4 dev clips first, then remaining clips)
    # =========================================================================
    start_b_dt = datetime.datetime.now()
    print(f"\n>>> Starting STAGE B (Objects on GPU) at {start_b_dt.strftime('%H:%M:%S')} IST")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"\n### Batch Stage B (Objects) Started: {start_b_dt.strftime('%Y-%m-%d %H:%M:%S')} IST\n")

    # Sort: 4 dev clips first
    dev_names = config.get("dev_clips", [])
    dev_clips = []
    other_clips = []
    for c in clips:
        # Match dev stems (handle 60-006 / 60-008 alias)
        is_dev = False
        for d in dev_names:
            if d in c.stem or (d.endswith("006") and c.stem.endswith("008")) or (d.endswith("008") and c.stem.endswith("006")):
                is_dev = True
                break
        if is_dev:
            dev_clips.append(c)
        else:
            other_clips.append(c)

    ordered_clips = dev_clips + other_clips
    print(f"Order: {len(dev_clips)} dev clips first, then {len(other_clips)} other clips")

    t_start_b = time.perf_counter()
    stage_b_done_count = 0
    for idx, c in enumerate(ordered_clips, 1):
        stem = c.stem
        is_dev_clip = (idx <= len(dev_clips))
        # Dev clips are force-run to ensure the latest exact classes and ignore list are applied
        res = run_stage_b_clip(c, out_dir, config=config, step=2, force=is_dev_clip)
        if res["status"] == "skipped":
            print(f"[{idx}/40] Stage B SKIPPED (already complete): {stem}")
        else:
            stage_b_done_count += 1
            line = f"- [Stage B {idx:02d}/40] `{stem}`: {res['time']:.1f}s ({res['time']/60:.2f}m) | {res['fps']:.1f} FPS | {res['frames']} frames\n"
            print(f"[{idx}/40] Stage B DONE: {stem} in {res['time']:.1f}s ({res['fps']:.1f} FPS)")
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(line)

    elap_b = time.perf_counter() - t_start_b
    print(f">>> STAGE B COMPLETED in {elap_b/60:.2f} min ({stage_b_done_count} clips processed, {len(ordered_clips)-stage_b_done_count} skipped)")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"- Stage B Finished: {datetime.datetime.now().strftime('%H:%M:%S')} IST (processed {stage_b_done_count} clips in {elap_b/60:.2f} min)\n")


if __name__ == "__main__":
    main()


"""Main pipeline runner for EgoHack egocentric vision auto-annotation.

Usage:
  python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20
  python run_pipeline.py --all --excerpt 20
  python run_pipeline.py --clip MTL-SGF00106-036
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import yaml

from common.io import frame_generator
from common.validate import validate_hands_jsonl, validate_objects_jsonl, validate_contacts_frames_jsonl
from common.schema import FrameObjects, FrameContacts
from hands.detect import HandDetector
from hands.metrics import calculate_metrics
from objects.detector import ObjectDetector, get_scene_vocabulary_for_clip
from contacts.baseline import ContactEstimator, run_contact_stage_from_files
from contacts.timeline import generate_contacts_csv
from viz.render import render_frame_hands, ensure_h264_compatible


def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_clip_path(clip_arg: str, config: dict) -> Path:
    clips_dir = Path(config.get("data", {}).get("clips_dir", "data/clips"))
    raw_dir = Path(config.get("data", {}).get("raw_dir", "data/raw"))

    stem = Path(clip_arg).stem
    candidates = [
        clips_dir / f"{stem}.mp4",
        raw_dir / f"{stem}.mp4",
        Path(clip_arg),
    ]
    dev_base = Path("C:/egohack_data")
    if dev_base.exists():
        for d in dev_base.iterdir():
            if d.is_dir() and "clip" in d.name:
                candidates.append(d / f"{stem}.mp4")

    # Handle common alias: 000060-006 <-> 000060-008
    if "000060-006" in stem:
        candidates.append(clips_dir / "mytronlabs-000060-008.mp4")
    elif "000060-008" in stem:
        candidates.append(clips_dir / "mytronlabs-000060-006.mp4")

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    raise FileNotFoundError(f"Could not locate clip for '{clip_arg}'. Checked: {[str(c) for c in candidates]}")


def run_clip_pipeline(
    clip_path: Path,
    out_root: Path,
    step: int = 2,
    excerpt_sec: Optional[float] = None,
    force: bool = False,
    render_video: bool = True,
    stage: str = "all",
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    clip_stem = clip_path.stem
    clip_out_dir = out_root / clip_stem
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    hands_jsonl_path = clip_out_dir / "hands.jsonl"
    objects_jsonl_path = clip_out_dir / "objects.jsonl"
    contacts_frames_jsonl_path = clip_out_dir / "contacts_frames.jsonl"
    contacts_csv_path = clip_out_dir / "contacts.csv"
    metrics_json_path = clip_out_dir / "metrics_hands.json"
    overlay_video_path = clip_out_dir / "overlay.mp4"

    if config is None:
        config = load_config()

    if stage == "contact":
        if not hands_jsonl_path.exists() or not objects_jsonl_path.exists():
            raise FileNotFoundError(f"Cannot run contact stage without {hands_jsonl_path} and {objects_jsonl_path}")
        print(f"[{clip_stem}] Running standalone Contact stage from existing hands & objects...")
        n_c = run_contact_stage_from_files(
            hands_jsonl_path=hands_jsonl_path,
            objects_jsonl_path=objects_jsonl_path,
            contacts_jsonl_path=contacts_frames_jsonl_path,
            contacts_csv_path=contacts_csv_path,
            clip_name=clip_stem,
            config=config,
        )
        print(f"[{clip_stem}] Contact stage complete: {n_c} frames written to {contacts_csv_path}")
        return {"contacts_frames": n_c, "clip": clip_stem}


    # Stage skip check
    all_exist = (
        hands_jsonl_path.exists() and
        objects_jsonl_path.exists() and
        contacts_frames_jsonl_path.exists() and
        contacts_csv_path.exists() and
        metrics_json_path.exists()
    )
    if not force and all_exist:
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                old_m = json.load(f)
            old_frames = old_m.get("total_frames_processed", 0)
            # If running full clip (excerpt_sec is None) but old run was only an excerpt (<1000 frames), rerun
            if excerpt_sec is None and old_frames < 1000:
                pass
            elif not render_video or overlay_video_path.exists():
                print(f"[{clip_stem}] All outputs already exist in {clip_out_dir}, skipping (use --force to rerun).")
                return old_m
        except Exception:
            pass

    print(f"\n=======================================================")
    print(f"Processing Clip: {clip_stem}")
    print(f"Input: {clip_path} | Excerpt: {excerpt_sec}s | Step: {step}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(str(clip_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if excerpt_sec is not None:
        max_frames_to_read = int(round(excerpt_sec * video_fps))
    else:
        max_frames_to_read = total_frames

    detector = HandDetector(
        model_path="weights/hand_landmarker.task",
        num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        filter_cutoff=1.0,
        filter_beta=0.01,
    )

    obj_cfg = config.get("objects", {})
    scene_classes, scene_fixed = get_scene_vocabulary_for_clip(clip_stem, config)
    ignore_classes = set(obj_cfg.get("ignore_classes", []))

    obj_detector = ObjectDetector(
        model_name=obj_cfg.get("model", "yolov8s-worldv2.pt"),
        classes=scene_classes,
        conf_threshold=obj_cfg.get("confidence_threshold", 0.10),
        crop_conf_threshold=obj_cfg.get("crop_confidence_threshold", 0.06),
        hand_iou_threshold=obj_cfg.get("hand_object_iou_threshold", 0.6),
        fixed_classes=scene_fixed,
        ignore_classes=ignore_classes,
    )

    contact_cfg = obj_cfg.get("contact", {})
    contact_estimator = ContactEstimator(
        fixed_classes=scene_fixed,
        expansion_ratio=contact_cfg.get("expansion_ratio", 0.15),
        min_fingertips=contact_cfg.get("min_fingertips", 2),
        min_held_frames=contact_cfg.get("min_held_frames", 2),
        min_hand_obj_iou=contact_cfg.get("min_hand_obj_iou", 0.08),
        ignore_classes=ignore_classes,
    )

    # Setup video writer
    writer = None
    if render_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_video_path), fourcc, video_fps, (width, height))

    frames_read = 0
    step_frames_count = 0
    wearer_det_count = 0
    wearer_dual_count = 0
    frames_with_objs = 0
    frames_gained_obj = 0
    total_det_time = 0.0

    hands_jsonl_lines = []
    objects_jsonl_lines = []
    contacts_jsonl_lines = []
    last_frame_objects: Optional[FrameObjects] = None
    t_start_total = time.perf_counter()

    for frame_idx, t_sec, frame in frame_generator(clip_path, step=1, max_frames=max_frames_to_read):
        frames_read += 1
        is_step_frame = (frame_idx % step == 0)

        if is_step_frame:
            step_frames_count += 1
            t0 = time.perf_counter()
            frame_hands = detector.process_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                width=width,
                height=height,
            )
            wearer_hands = [h for h in frame_hands.hands if h.visible and h.is_wearer]
            if len(wearer_hands) >= 1:
                wearer_det_count += 1
            if len(wearer_hands) >= 2:
                wearer_dual_count += 1

            # Detect objects
            wearer_bboxes = [h.bbox for h in wearer_hands]
            objs, gained = obj_detector.detect_frame(frame, wearer_bboxes)
            if gained:
                frames_gained_obj += 1
            if len(objs) > 0:
                frames_with_objs += 1

            frame_objects = FrameObjects(frame=frame_idx, t=t_sec, objects=objs)
            last_frame_objects = frame_objects

            # Estimate contact
            frame_contacts = contact_estimator.process_frame(
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                wearer_hands=frame_hands.hands,
                detected_objects=objs,
            )
            dt = time.perf_counter() - t0
            total_det_time += dt
        else:
            # Propagate forward with 1-Euro smoother and velocity prediction
            frame_hands = detector.process_skipped_frame(
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
            )
            objs = last_frame_objects.objects if last_frame_objects else []
            frame_objects = FrameObjects(frame=frame_idx, t=t_sec, objects=objs)

            # Estimate contact
            frame_contacts = contact_estimator.process_frame(
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                wearer_hands=frame_hands.hands,
                detected_objects=objs,
            )

        hands_jsonl_lines.append(frame_hands.to_dict())
        objects_jsonl_lines.append(frame_objects.to_dict())
        contacts_jsonl_lines.append(frame_contacts.to_dict())

        # Render overlay
        if writer is not None:
            running_det = wearer_det_count / max(1, step_frames_count)
            running_dual = wearer_dual_count / max(1, step_frames_count)
            inst_fps = step_frames_count / max(1e-4, total_det_time)
            viz_frame = render_frame_hands(
                frame=frame,
                frame_hands=frame_hands,
                running_det_share=running_det,
                running_dual_share=running_dual,
                current_fps=inst_fps,
                frame_objects=frame_objects,
                frame_contacts=frame_contacts,
            )
            writer.write(viz_frame)

    t_end_total = time.perf_counter()
    side_flip_log = list(detector.tracker.side_flip_log)
    detector.close()

    if writer is not None:
        writer.release()
        ensure_h264_compatible(overlay_video_path)

    # 1. Write hands.jsonl
    with open(hands_jsonl_path, "w", encoding="utf-8") as f:
        for item in hands_jsonl_lines:
            f.write(json.dumps(item) + "\n")

    # 2. Write objects.jsonl
    with open(objects_jsonl_path, "w", encoding="utf-8") as f:
        for item in objects_jsonl_lines:
            f.write(json.dumps(item) + "\n")

    # 3. Write contacts_frames.jsonl
    with open(contacts_frames_jsonl_path, "w", encoding="utf-8") as f:
        for item in contacts_jsonl_lines:
            f.write(json.dumps(item) + "\n")

    # 4. Write contacts.csv
    intervals = generate_contacts_csv(clip_stem, contacts_frames_jsonl_path, contacts_csv_path)

    # Validate outputs against schema
    valid_hands, errs_hands = validate_hands_jsonl(hands_jsonl_path)
    valid_objs, errs_objs = validate_objects_jsonl(objects_jsonl_path)
    valid_contacts, errs_contacts = validate_contacts_frames_jsonl(contacts_frames_jsonl_path)

    valid_all = valid_hands and valid_objs and valid_contacts
    print(f"Validation: hands={valid_hands}, objects={valid_objs}, contacts_frames={valid_contacts}")
    if not valid_all:
        print(f"Schema warnings: {errs_hands[:3]} | {errs_objs[:3]} | {errs_contacts[:3]}")

    det_fps = step_frames_count / max(1e-4, total_det_time)
    pipeline_fps = frames_read / max(1e-4, t_end_total - t_start_total)

    metrics = calculate_metrics(
        hands_jsonl_path=hands_jsonl_path,
        fps=video_fps,
        eval_step=step,
        det_fps=det_fps,
        total_fps=pipeline_fps,
        side_flip_log=side_flip_log,
    )

    # Object and contact metrics
    step_indices = [i for i in range(len(hands_jsonl_lines)) if i % step == 0]
    fw_wearer = 0
    fw_obj_nearby = 0
    fw_contact = 0
    class_counts: Dict[str, int] = {}
    contact_states: Dict[str, int] = {"none": 0, "portable": 0, "fixed": 0}

    for idx in step_indices:
        h_f = hands_jsonl_lines[idx]
        o_f = objects_jsonl_lines[idx]
        c_f = contacts_jsonl_lines[idx]

        w_hands = [h for h in h_f.get("hands", []) if h.get("visible", False) and h.get("is_wearer", True)]
        objs = o_f.get("objects", [])
        conts = [c for c in c_f.get("contacts", []) if c.get("state", "none") != "none"]

        if len(w_hands) > 0:
            fw_wearer += 1
            if len(objs) > 0:
                fw_obj_nearby += 1
            if len(conts) > 0:
                fw_contact += 1

        for o in objs:
            lbl = o.get("label", "unknown")
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        for c in c_f.get("contacts", []):
            st = c.get("state", "none")
            contact_states[st] = contact_states.get(st, 0) + 1

    top_5 = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    share_obj = (fw_obj_nearby / fw_wearer * 100.0) if fw_wearer > 0 else 0.0
    share_cont = (fw_contact / fw_wearer * 100.0) if fw_wearer > 0 else 0.0

    metrics["objects_jsonl"] = str(objects_jsonl_path)
    metrics["contacts_frames_jsonl"] = str(contacts_frames_jsonl_path)
    metrics["contacts_csv"] = str(contacts_csv_path)
    metrics["overlay_video"] = str(overlay_video_path)
    metrics["hands_jsonl"] = str(hands_jsonl_path)
    metrics["schema_valid"] = valid_all
    metrics["frames_with_wearer_hand"] = fw_wearer
    metrics["frames_with_obj_nearby"] = fw_obj_nearby
    metrics["pct_wearer_with_obj_nearby"] = round(share_obj, 1)
    metrics["frames_with_contact"] = fw_contact
    metrics["pct_wearer_with_contact"] = round(share_cont, 1)
    metrics["top_5_object_classes"] = top_5
    metrics["contact_states"] = contact_states
    metrics["frames_gained_obj_crop"] = frames_gained_obj

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- Clip {clip_stem} Metrics ---")
    print(f"  Frames evaluated: {metrics['frames_evaluated_detection']} detection frames")
    print(f"  Wearer single hand share: {metrics['share_wearer_any_hand']*100:.1f}%")
    print(f"  Wearer dual hands share: {metrics['share_wearer_both_hands']*100:.1f}%")
    print(f"  Wearer frames with nearby object: {fw_obj_nearby}/{fw_wearer} ({share_obj:.1f}%)")
    print(f"  Wearer frames with contact: {fw_contact}/{fw_wearer} ({share_cont:.1f}%)")
    print(f"  Top 5 object classes: {top_5}")
    print(f"  Frames gaining object via crop pass: {frames_gained_obj}")
    print(f"  End-to-end Pipeline FPS: {pipeline_fps:.2f} FPS (Det+Obj+Contact FPS: {det_fps:.2f})")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EgoHack Hand Tracking Pipeline")
    parser.add_argument("--clip", type=str, default=None, help="Clip name or path to process")
    parser.add_argument("--all", action="store_true", help="Process all available clips")
    parser.add_argument("--excerpt", type=float, default=None, help="Excerpt duration in seconds (e.g. 20)")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "hands", "objects", "contact"], help="Pipeline stage to run")
    parser.add_argument("--step", type=int, default=2, help="Detection stride (default: 2)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    parser.add_argument("--no-viz", action="store_true", help="Disable overlay video generation")

    args = parser.parse_args()
    config = load_config()

    out_root = Path(config.get("data", {}).get("out_dir", "out"))
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        clips_dir = Path(config.get("data", {}).get("clips_dir", "data/clips"))
        if not clips_dir.exists() and Path("C:/egohack_data").exists():
            for d in Path("C:/egohack_data").iterdir():
                if d.is_dir() and "clip" in d.name:
                    clips_dir = d
                    break
        clips = sorted(list(clips_dir.glob("*.mp4")))
        if not clips:
            print(f"No clips found in {clips_dir}")
            sys.exit(1)
        print(f"Running pipeline on ALL {len(clips)} clips...")
        all_metrics = []
        for c in clips:
            m = run_clip_pipeline(
                clip_path=c,
                out_root=out_root,
                step=args.step,
                excerpt_sec=args.excerpt,
                force=args.force,
                render_video=not args.no_viz,
                stage=args.stage,
                config=config,
            )
            all_metrics.append(m)
        print("\nAll clips completed successfully!")
    elif args.clip:
        resolved = resolve_clip_path(args.clip, config)
        run_clip_pipeline(
            clip_path=resolved,
            out_root=out_root,
            step=args.step,
            excerpt_sec=args.excerpt,
            force=args.force,
            render_video=not args.no_viz,
            stage=args.stage,
            config=config,
        )
    else:
        print("Please provide --clip <name> or --all. Example:")
        print("  python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20")
        sys.exit(1)


if __name__ == "__main__":
    main()


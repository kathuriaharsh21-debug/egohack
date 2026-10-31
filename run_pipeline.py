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
from common.validate import validate_hands_jsonl
from hands.detect import HandDetector
from hands.metrics import calculate_metrics
from viz.render import render_frame_hands, ensure_h264_compatible


def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_clip_path(clip_arg: str, config: dict) -> Path:
    clips_dir = Path(config.get("data", {}).get("clips_480p_dir", "data/clips_480p"))
    raw_dir = Path(config.get("data", {}).get("clips_dir", "data/clips"))

    stem = Path(clip_arg).stem
    candidates = [
        clips_dir / f"{stem}.mp4",
        raw_dir / f"{stem}.mp4",
        Path(clip_arg),
    ]

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
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    clip_stem = clip_path.stem
    clip_out_dir = out_root / clip_stem
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    hands_jsonl_path = clip_out_dir / "hands.jsonl"
    metrics_json_path = clip_out_dir / "metrics_hands.json"
    overlay_video_path = clip_out_dir / "overlay.mp4"

    # Stage skip check
    if not force and hands_jsonl_path.exists() and metrics_json_path.exists():
        if not render_video or overlay_video_path.exists():
            print(f"[{clip_stem}] Outputs already exist in {clip_out_dir}, skipping (use --force to rerun).")
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

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

    # Setup video writer
    writer = None
    if render_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_video_path), fourcc, video_fps, (width, height))

    frames_read = 0
    step_frames_count = 0
    wearer_det_count = 0
    wearer_dual_count = 0
    total_det_time = 0.0

    jsonl_lines = []
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
            dt = time.perf_counter() - t0
            total_det_time += dt

            wearer_hands = [h for h in frame_hands.hands if h.visible and h.is_wearer]
            if len(wearer_hands) >= 1:
                wearer_det_count += 1
            if len(wearer_hands) >= 2:
                wearer_dual_count += 1
        else:
            # Propagate forward with 1-Euro smoother and velocity prediction
            frame_hands = detector.process_skipped_frame(
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
            )

        jsonl_lines.append(frame_hands.to_dict())

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
            )
            writer.write(viz_frame)

    t_end_total = time.perf_counter()
    side_flip_log = list(detector.tracker.side_flip_log)
    detector.close()

    if writer is not None:
        writer.release()
        ensure_h264_compatible(overlay_video_path)

    # Write hands.jsonl
    with open(hands_jsonl_path, "w", encoding="utf-8") as f:
        for item in jsonl_lines:
            f.write(json.dumps(item) + "\n")

    # Validate output
    valid, errors = validate_hands_jsonl(hands_jsonl_path)
    if not valid:
        print(f"WARNING: Schema validation failed for {hands_jsonl_path}: {errors[:5]}")
    else:
        print(f"Validation: PASSED ({len(jsonl_lines)} frames written to {hands_jsonl_path.name})")

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
    metrics["overlay_video"] = str(overlay_video_path)
    metrics["hands_jsonl"] = str(hands_jsonl_path)
    metrics["schema_valid"] = valid

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- Clip {clip_stem} Metrics ---")
    print(f"  Frames evaluated: {metrics['frames_evaluated_detection']} detection frames")
    print(f"  Wearer single hand share: {metrics['share_wearer_any_hand']*100:.1f}%")
    print(f"  Wearer dual hands share: {metrics['share_wearer_both_hands']*100:.1f}%")
    print(f"  All hands share (incl bystanders): {metrics['share_all_any_hand']*100:.1f}%")
    print(f"  Total unique tracks: {metrics['total_unique_tracks']} | ID switches: {metrics['id_switches']}")
    print(f"  Side swaps logged: {metrics['side_swaps_logged']}")
    print(f"  Mean landmark jitter: {metrics['mean_landmark_jitter_px']} px")
    print(f"  Detector FPS: {metrics['detector_fps']} FPS | Pipeline FPS: {metrics['total_pipeline_fps']} FPS")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EgoHack Hand Tracking Pipeline")
    parser.add_argument("--clip", type=str, default=None, help="Clip name or path to process")
    parser.add_argument("--all", action="store_true", help="Process all available clips")
    parser.add_argument("--excerpt", type=float, default=None, help="Excerpt duration in seconds (e.g. 20)")
    parser.add_argument("--step", type=int, default=2, help="Detection stride (default: 2)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    parser.add_argument("--no-viz", action="store_true", help="Disable overlay video generation")

    args = parser.parse_args()
    config = load_config()

    out_root = Path(config.get("data", {}).get("out_dir", "out"))
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        clips_dir = Path(config.get("data", {}).get("clips_480p_dir", "data/clips_480p"))
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
            config=config,
        )
    else:
        print("Please provide --clip <name> or --all. Example:")
        print("  python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20")
        sys.exit(1)


if __name__ == "__main__":
    main()

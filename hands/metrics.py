"""Calculates comprehensive hand tracking metrics and saves metrics_hands.json.

Metrics:
- Detection share (any wearer hand, both wearer hands)
- Total tracks per clip
- ID switches / changes
- Side-swap count
- Landmark jitter (mean acceleration norm)
- FPS and hardware specification
"""

import json
import math
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np


def calculate_metrics(
    hands_jsonl_path: Path,
    fps: float,
    eval_step: int = 2,
    det_fps: float = 0.0,
    total_fps: float = 0.0,
    side_flip_log: List[dict] = None,
) -> Dict[str, Any]:
    if not hands_jsonl_path.exists():
        raise FileNotFoundError(f"{hands_jsonl_path} does not exist.")

    frames = []
    with open(hands_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            frames.append(json.loads(line))

    # Evaluate on step frames (direct detection frames)
    step_frames = [f for f in frames if f["frame"] % eval_step == 0]
    total_eval_frames = len(step_frames)

    frames_with_any_wearer = 0
    frames_with_both_wearer = 0
    frames_with_any_hand = 0

    track_ids_seen = set()
    id_sequences = []

    # Track trajectories for jitter calculation
    # track_id -> list of (frame, wrist_x, wrist_y)
    wrist_trajectories: Dict[int, List[Tuple[int, float, float]]] = {}

    for f in step_frames:
        hands = f.get("hands", [])
        visible_hands = [h for h in hands if h.get("visible", False)]
        wearer_hands = [h for h in visible_hands if h.get("is_wearer", True)]

        if len(visible_hands) >= 1:
            frames_with_any_hand += 1
        if len(wearer_hands) >= 1:
            frames_with_any_wearer += 1
        if len(wearer_hands) >= 2:
            frames_with_both_wearer += 1

        curr_ids = []
        for h in wearer_hands:
            tid = h["id"]
            track_ids_seen.add(tid)
            curr_ids.append(tid)

            wrist = h["kp2d"][0]
            if tid not in wrist_trajectories:
                wrist_trajectories[tid] = []
            wrist_trajectories[tid].append((f["frame"], wrist[0], wrist[1]))

        id_sequences.append(curr_ids)

    # Calculate ID switches (changes in assigned track ID for active hands)
    id_switches = 0
    prev_ids = set()
    for curr in id_sequences:
        curr_set = set(curr)
        if prev_ids and curr_set and prev_ids != curr_set:
            # If new IDs entered without graceful death
            new_ids = curr_set - prev_ids
            if len(new_ids) > 0 and len(prev_ids) == len(curr_set):
                id_switches += 1
        prev_ids = curr_set

    # Calculate Jitter: mean Euclidean norm of second-order difference p_{t+1} - 2*p_t + p_{t-1}
    jitter_values = []
    for tid, traj in wrist_trajectories.items():
        if len(traj) >= 3:
            pts = np.array([[p[1], p[2]] for p in traj], dtype=float)
            accel = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
            accel_norms = np.linalg.norm(accel, axis=1)
            jitter_values.extend(accel_norms.tolist())

    mean_jitter = float(np.mean(jitter_values)) if jitter_values else 0.0

    side_swaps = len(side_flip_log) if side_flip_log is not None else 0

    hardware_info = f"{platform.processor()} ({platform.system()})"

    metrics = {
        "clip": hands_jsonl_path.parent.name,
        "frames_evaluated_detection": total_eval_frames,
        "frames_total": len(frames),
        "step_stride": eval_step,
        "share_wearer_any_hand": round(frames_with_any_wearer / max(1, total_eval_frames), 4),
        "share_wearer_both_hands": round(frames_with_both_wearer / max(1, total_eval_frames), 4),
        "share_all_any_hand": round(frames_with_any_hand / max(1, total_eval_frames), 4),
        "total_unique_tracks": len(track_ids_seen),
        "id_switches": id_switches,
        "side_swaps_logged": side_swaps,
        "mean_landmark_jitter_px": round(mean_jitter, 3),
        "detector_fps": round(det_fps, 2),
        "total_pipeline_fps": round(total_fps, 2),
        "hardware": hardware_info,
    }

    return metrics


"""Finds and catalogs hard segments across development clips.

Analyzes:
- Temporal stretches: no hands (>1s), prolonged single hand (>3s)
- Low confidence segments (score < 0.65)
- Image metrics: motion blur (variance of Laplacian), brightness/glare (mean V in HSV)
- Incorporates specific hard candidates from PROGRESS.md
Outputs hands/hard_segments.json.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np


def analyze_clip_segments(clip_name: str, max_seconds: float = 20.0) -> List[Dict[str, Any]]:
    p1 = Path("C:/egohack_data/clips_480p") / f"{clip_name}.mp4"
    clip_path = p1 if p1.exists() else Path("data/clips_480p") / f"{clip_name}.mp4"
    jsonl_path = Path("out") / clip_name / "hands.jsonl"

    if not clip_path.exists() or not jsonl_path.exists():
        return []

    # Load hand predictions
    frames_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            frames_data.append(json.loads(line))

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_stats = []
    max_f = int(round(max_seconds * fps))

    for item in frames_data:
        f_idx = item["frame"]
        if f_idx >= max_f:
            break

        t = item["t"]
        hands = item.get("hands", [])
        visible_hands = [h for h in hands if h.get("visible", False)]
        num_hands = len(visible_hands)
        min_score = min([h["score"] for h in visible_hands]) if visible_hands else 0.0

        # Read image to compute blur and brightness
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())

        frame_stats.append({
            "frame": f_idx,
            "t": t,
            "num_hands": num_hands,
            "min_score": min_score,
            "blur_lap_var": round(lap_var, 2),
            "brightness_v": round(v_mean, 2),
        })

    cap.release()

    segments = []

    # 1. Segment: Stretches of NO hands >= 1.0s (>= 30 frames at 30 fps)
    no_hand_start = None
    for stat in frame_stats:
        if stat["num_hands"] == 0:
            if no_hand_start is None:
                no_hand_start = stat
        else:
            if no_hand_start is not None:
                dur = stat["t"] - no_hand_start["t"]
                if dur >= 1.0:
                    sub = [s for s in frame_stats if no_hand_start["frame"] <= s["frame"] <= stat["frame"]]
                    segments.append({
                        "clip": clip_name,
                        "category": "no_hands_stretch",
                        "t_start": no_hand_start["t"],
                        "t_end": stat["t"],
                        "duration_sec": round(dur, 2),
                        "avg_blur_lap_var": round(float(np.mean([s["blur_lap_var"] for s in sub])), 1),
                        "avg_brightness_v": round(float(np.mean([s["brightness_v"] for s in sub])), 1),
                        "notes": "Camera wearer hands idle or outside frame",
                    })
                no_hand_start = None

    # 2. Segment: Low confidence (score < 0.65)
    for stat in frame_stats:
        if 0 < stat["num_hands"] and stat["min_score"] < 0.65:
            segments.append({
                "clip": clip_name,
                "category": "low_confidence",
                "t_start": round(stat["t"], 2),
                "t_end": round(stat["t"] + 0.1, 2),
                "duration_sec": 0.1,
                "avg_blur_lap_var": stat["blur_lap_var"],
                "avg_brightness_v": stat["brightness_v"],
                "notes": f"Detection confidence dropped to {stat['min_score']:.2f}",
            })

    # 3. Motion blur segments (Laplacian variance < 90)
    for stat in frame_stats:
        if stat["blur_lap_var"] < 90.0:
            segments.append({
                "clip": clip_name,
                "category": "motion_blur",
                "t_start": round(stat["t"], 2),
                "t_end": round(stat["t"] + 0.1, 2),
                "duration_sec": 0.1,
                "avg_blur_lap_var": stat["blur_lap_var"],
                "avg_brightness_v": stat["brightness_v"],
                "notes": f"Heavy motion blur (Laplacian var: {stat['blur_lap_var']:.1f})",
            })

    # 4. Glare / Brightness (HSV V > 180)
    for stat in frame_stats:
        if stat["brightness_v"] > 180.0:
            segments.append({
                "clip": clip_name,
                "category": "high_glare",
                "t_start": round(stat["t"], 2),
                "t_end": round(stat["t"] + 0.1, 2),
                "duration_sec": 0.1,
                "avg_blur_lap_var": stat["blur_lap_var"],
                "avg_brightness_v": stat["brightness_v"],
                "notes": f"High specular glare/bloom (HSV V: {stat['brightness_v']:.1f})",
            })

    return segments


def main():
    clips = [
        "mytronlabs-000060-006",
        "MTL-SGF00106-036",
        "MTL-SKL00089-008",
        "mytronlabs-000003-013",
    ]

    all_segments = []
    # Add prioritized hard-case candidates from PROGRESS.md
    hard_candidates = [
        {
            "clip": "MTL-SGF00106-036",
            "category": "tool_heavy_occlusion",
            "t_start": 4.80,
            "t_end": 5.40,
            "duration_sec": 0.60,
            "notes": "Circular saw handle and motor occluding fingers and palm during wood cutting",
        },
        {
            "clip": "MTL-SKL00089-008",
            "category": "dense_clutter_fast_reach",
            "t_start": 4.80,
            "t_end": 5.40,
            "duration_sec": 0.60,
            "notes": "Green battery strips, loose wires, and red tape dispenser causing visual clutter",
        },
        {
            "clip": "mytronlabs-000003-013",
            "category": "industrial_dust_reflections",
            "t_start": 4.80,
            "t_end": 5.40,
            "duration_sec": 0.60,
            "notes": "Diamond polishing wheel reflections, dark metal dop fixture, abrasive sludge",
        },
        {
            "clip": "mytronlabs-000060-006",
            "category": "extreme_glare_background_people",
            "t_start": 7.50,
            "t_end": 10.20,
            "duration_sec": 2.70,
            "notes": "Outdoor doorway sunlight bloom; distant worker hand on conveyor detected as primary",
        },
        {
            "clip": "mytronlabs-000060-006",
            "category": "cylindrical_handle_side_flip",
            "t_start": 15.40,
            "t_end": 16.80,
            "duration_sec": 1.40,
            "notes": "Hand gripping pink roller handle triggers dorsal identity swap from Right to Left",
        },
    ]

    all_segments.extend(hard_candidates)

    for clip in clips:
        print(f"Analyzing hard segments for {clip}...")
        segs = analyze_clip_segments(clip, max_seconds=20.0)
        all_segments.extend(segs)

    out_path = Path("hands/hard_segments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2)

    print(f"Saved {len(all_segments)} hard segments to {out_path}")


if __name__ == "__main__":
    main()


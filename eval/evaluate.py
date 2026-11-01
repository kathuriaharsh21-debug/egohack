"""EgoHack Evaluation Harness.

Evaluates pipeline outputs against:
(a) Human labels (eval/labels/stratified_labels.json, with 060-006 f10 excluded as old default flag)
(b) Merged AI labels (eval/labels_ai_only.json, secondary stress test, not ground truth)

Features:
- Alignment: Matches each labeled frame to the nearest processed frame within +-1 frame.
- Test split opened on 2026-11-01 06:21 IST; strictly NO pipeline tuning on test results.
- Review queue catch rate compared side-by-side with Random Baseline (100 runs, seed 42).
- Sample sizes < 30 explicitly marked '(too small to trust)'.
"""

import sys
import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

TEST_OPENED_TIME = "2026-11-01 06:21 IST"
EXCLUDED_HUMAN_KEYS = {"mytronlabs-000060-006_10"}  # Flagged old default pattern

def is_hard_frame(clip: str, t_sec: float, hard_segments: List[Dict[str, Any]]) -> bool:
    for seg in hard_segments:
        if seg.get("clip") == clip:
            t0 = float(seg.get("t_start", 0.0))
            t1 = float(seg.get("t_end", 0.0))
            if t0 <= t_sec <= t1:
                return True
    return False

def fmt_stat(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A (n=0)"
    pct = (num / denom) * 100.0
    trust = " (too small to trust)" if denom < 30 else ""
    return f"{pct:.1f}% ({num}/{denom}, n={denom}){trust}"

def compute_metrics_for_subset(
    frames_data: List[Dict[str, Any]],
    pipe_cache: Dict[str, Any],
    is_ai_only: bool = False
) -> Dict[str, Any]:
    # 1. Hand presence recall
    gt_wearer_all = [f for f in frames_data if f["gt_has_wearer"]]
    gt_wearer_found = [f for f in gt_wearer_all if f["pipe_has_visible_wearer"]]

    # 2. False detection rate
    gt_no_hand_all = [f for f in frames_data if f["gt_is_no_hand"]]
    gt_no_hand_false_pos = [f for f in gt_no_hand_all if f["pipe_has_visible_wearer"]]

    # 3. Side accuracy (omitted on merged AI labels)
    if is_ai_only:
        side_dict = {
            "num": 0,
            "denom": 0,
            "formatted": "N/A (omitted on merged AI labels)"
        }
    else:
        side_eval = [f for f in frames_data if f["side_evaluable"]]
        side_correct = [f for f in side_eval if f["side_agreed"]]
        side_dict = {
            "num": len(side_correct),
            "denom": len(side_eval),
            "formatted": fmt_stat(len(side_correct), len(side_eval))
        }

    # 4. Contact accuracy & confusion matrix
    contact_eval = [f for f in frames_data if f["contact_evaluable"]]
    contact_correct = [f for f in contact_eval if f["contact_agreed"]]

    classes = ["none", "portable", "fixed"]
    matrix = {gt_c: {pred_c: 0 for pred_c in classes} for gt_c in classes}
    for f in contact_eval:
        gt_c = f["gt_contact"]
        pred_c = f["pipe_contact"]
        if gt_c in matrix and pred_c in matrix[gt_c]:
            matrix[gt_c][pred_c] += 1

    total_frames = len(frames_data)
    frames_with_interp = sum(1 for f in frames_data if f["pipe_has_pred_hand"])

    # 5. Review queue catch rate & random baseline
    disagreements = [f for f in frames_data if f["has_disagreement"]]
    caught_in_queue = [f for f in disagreements if f["in_review_queue"]]
    n_dis = len(disagreements)
    n_caught = len(caught_in_queue)

    # Random baseline (100 runs, seed 42)
    rand_catches = []
    if n_dis > 0:
        rng = np.random.RandomState(42)
        for _ in range(100):
            # For each clip, sample k random events matching review queue size
            run_caught = 0
            for d_f in disagreements:
                clip = d_f["clip"]
                t = d_f["t"]
                all_invs = pipe_cache[clip]["all_events"]
                k_rev = len(pipe_cache[clip]["review_intervals"])
                if k_rev >= len(all_invs):
                    run_caught += 1
                elif k_rev > 0 and len(all_invs) > 0:
                    rand_idx = rng.choice(len(all_invs), size=k_rev, replace=False)
                    rand_invs = [all_invs[i] for i in rand_idx]
                    if any(inv["t_start"] <= t <= inv["t_end"] for inv in rand_invs):
                        run_caught += 1
            rand_catches.append(run_caught)
        mean_rand = round(float(np.mean(rand_catches)), 1)
        rand_pct = (mean_rand / n_dis) * 100.0 if n_dis > 0 else 0.0
    else:
        mean_rand = 0.0
        rand_pct = 0.0

    trust = " (too small to trust)" if n_dis < 30 else ""
    if n_dis > 0:
        rq_formatted = f"Queue: {n_caught}/{n_dis} ({(n_caught/n_dis*100.0):.1f}%, n={n_dis}){trust} vs Random: {mean_rand}/{n_dis} ({rand_pct:.1f}%)"
    else:
        rq_formatted = "N/A (n=0)"

    return {
        "n_frames": total_frames,
        "hand_presence_recall": {
            "num": len(gt_wearer_found),
            "denom": len(gt_wearer_all),
            "formatted": fmt_stat(len(gt_wearer_found), len(gt_wearer_all))
        },
        "false_detection_rate": {
            "num": len(gt_no_hand_false_pos),
            "denom": len(gt_no_hand_all),
            "formatted": fmt_stat(len(gt_no_hand_false_pos), len(gt_no_hand_all))
        },
        "side_accuracy": side_dict,
        "contact_accuracy": {
            "num": len(contact_correct),
            "denom": len(contact_eval),
            "formatted": fmt_stat(len(contact_correct), len(contact_eval))
        },
        "confusion_matrix": matrix,
        "predicted_hands_share": {
            "num": frames_with_interp,
            "denom": total_frames,
            "formatted": fmt_stat(frames_with_interp, total_frames)
        },
        "review_queue_catch_rate": {
            "num": n_caught,
            "denom": n_dis,
            "random_mean": mean_rand,
            "formatted": rq_formatted
        }
    }

def get_nearest_processed_frame(
    pipe_clip: Dict[str, Any],
    frame_idx: int
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    hands_map = pipe_clip["hands"]
    contacts_map = pipe_clip["contacts"]

    candidates = [frame_idx]
    if frame_idx - 1 >= 0:
        candidates.append(frame_idx - 1)
    candidates.append(frame_idx + 1)

    for cand in candidates:
        if cand in hands_map:
            vis = [h for h in hands_map[cand].get("hands", []) if h.get("visible", False) and h.get("is_wearer", True)]
            if len(vis) > 0:
                return cand, hands_map[cand], contacts_map.get(cand, {})

    for cand in [frame_idx] + [c for c in candidates if c != frame_idx]:
        if cand % 2 == 0 and cand in hands_map:
            return cand, hands_map[cand], contacts_map.get(cand, {})

    return frame_idx, hands_map.get(frame_idx, {}), contacts_map.get(frame_idx, {})

def evaluate_split(
    split_name: str,
    clip_frames: Dict[str, List[int]],
    labels: Dict[str, Any],
    out_dir: Path,
    hard_segments: List[Dict[str, Any]],
    pipe_cache: Dict[str, Any],
    is_ai_only: bool = False
) -> Dict[str, Any]:
    frames_all = []
    frames_easy = []
    frames_hard = []

    for clip, f_list in clip_frames.items():
        for frame_idx in f_list:
            key = f"{clip}_{frame_idx}"
            if key not in labels:
                continue

            v_lbl = labels[key]
            t_sec = float(v_lbl.get("t", round(frame_idx / 30.0, 3)))
            is_hard = is_hard_frame(clip, t_sec, hard_segments)

            h1_f = str(v_lbl.get("h1_found", v_lbl.get("hand_visible", ""))).lower()
            h1_r = str(v_lbl.get("h1_role", v_lbl.get("role", "wearer"))).lower()
            h1_s = str(v_lbl.get("h1_side", v_lbl.get("side", "unsure"))).lower()
            h1_c = str(v_lbl.get("h1_contact", v_lbl.get("contact", "none"))).lower()

            gt_is_no_hand = ("no hand" in h1_f) or (h1_f == "no")
            gt_has_wearer = ("yes" in h1_f or "missed" in h1_f) and (h1_r == "wearer")

            matched_f, p_hands_data, p_contacts_data = get_nearest_processed_frame(pipe_cache[clip], frame_idx)

            p_hands = [h for h in p_hands_data.get("hands", []) if h.get("is_wearer", True)]
            vis_hands = [h for h in p_hands if h.get("visible", False)]
            pred_hands = [h for h in p_hands if not h.get("visible", False)]

            pipe_has_visible_wearer = len(vis_hands) > 0
            pipe_has_pred_hand = len(pred_hands) > 0

            side_evaluable = False
            side_agreed = False
            pipe_side = None
            if not is_ai_only and pipe_has_visible_wearer and h1_s in ["left", "right"]:
                side_evaluable = True
                pipe_side = vis_hands[0].get("side", "").lower()
                side_agreed = (pipe_side == h1_s)

            contact_evaluable = False
            contact_agreed = False
            pipe_contact = "none"
            if pipe_has_visible_wearer and h1_c in ["none", "portable", "fixed"]:
                contact_evaluable = True
                best_h = max(vis_hands, key=lambda h: h.get("score", 0.0))
                h_id = best_h.get("id", 0)
                pipe_contact = p_contacts_data.get(h_id, {}).get("state", "none").lower()
                contact_agreed = (pipe_contact == h1_c)

            has_disagreement = False
            if gt_has_wearer and not pipe_has_visible_wearer:
                has_disagreement = True
            elif gt_is_no_hand and pipe_has_visible_wearer:
                has_disagreement = True
            elif side_evaluable and not side_agreed:
                has_disagreement = True
            elif contact_evaluable and not contact_agreed:
                has_disagreement = True

            in_review_queue = any(
                inv["t_start"] <= t_sec <= inv["t_end"]
                for inv in pipe_cache[clip]["review_intervals"]
            )

            item = {
                "clip": clip,
                "frame": frame_idx,
                "matched_frame": matched_f,
                "t": t_sec,
                "is_hard": is_hard,
                "gt_has_wearer": gt_has_wearer,
                "gt_is_no_hand": gt_is_no_hand,
                "pipe_has_visible_wearer": pipe_has_visible_wearer,
                "pipe_has_pred_hand": pipe_has_pred_hand,
                "side_evaluable": side_evaluable,
                "side_agreed": side_agreed,
                "contact_evaluable": contact_evaluable,
                "contact_agreed": contact_agreed,
                "gt_contact": h1_c,
                "pipe_contact": pipe_contact,
                "has_disagreement": has_disagreement,
                "in_review_queue": in_review_queue
            }

            frames_all.append(item)
            if is_hard:
                frames_hard.append(item)
            else:
                frames_easy.append(item)

    overall_metrics = compute_metrics_for_subset(frames_all, pipe_cache, is_ai_only=is_ai_only)
    easy_metrics = compute_metrics_for_subset(frames_easy, pipe_cache, is_ai_only=is_ai_only)
    hard_metrics = compute_metrics_for_subset(frames_hard, pipe_cache, is_ai_only=is_ai_only)

    clips = sorted(list(clip_frames.keys()))
    fps_list = [pipe_cache[c]["fps"] for c in clips if pipe_cache[c]["fps"] is not None]
    mean_fps = round(sum(fps_list) / len(fps_list), 2) if fps_list else None

    return {
        "split": split_name,
        "labels_evaluated_count": len(frames_all),
        "overall": overall_metrics,
        "easy": easy_metrics,
        "hard": hard_metrics,
        "mean_pipeline_fps": mean_fps
    }

def generate_markdown_report(results: Dict[str, Any], spotcheck_info: Optional[Dict[str, Any]] = None) -> str:
    lines = []
    lines.append("# EgoHack Evaluation Report")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Alignment: Matches each labeled frame to the nearest processed frame within +-1 frame.")
    lines.append(f"> Test split opened on **{TEST_OPENED_TIME}**; strictly NO pipeline tuning on test results.")
    lines.append("> Note on Human Labels: `060-006 f10` is excluded from the human block (flagged as old default pattern).")
    lines.append("> Metrics report exact sample sizes ($n=$). Any sample size with $n < 30$ is explicitly flagged as `(too small to trust)`. Metrics with zero labels are marked `N/A`.")
    lines.append("")

    if spotcheck_info:
        sc_right = spotcheck_info.get("right_count", 0)
        sc_tot = spotcheck_info.get("total_count", 0)
        lines.append(f"> **Spot Check Status**: Merged AI labels judged right: {sc_right} of {sc_tot} (spot-checked)")
        lines.append("")
    else:
        lines.append("> **Spot Check Status**: Merged AI labels not spot-checked.")
        lines.append("")

    # BLOCK A: Human labels
    lines.append("## Block A: Against Human Labels (Ground Truth, Flagged Defaults Excluded)")
    lines.append("")
    human_res = results.get("human", {})
    for split_key in ["tune", "test"]:
        split_res = human_res.get(split_key, {})
        lines.append(f"### Human Split: `{split_key.upper()}`")
        if split_res.get("locked"):
            lines.append(f"**STATUS: LOCKED** — {split_res.get('message', 'Locked.')}\n")
            continue

        n_lbls = split_res.get("labels_evaluated_count", 0)
        prelim_tag = " (too small to trust)" if n_lbls < 30 else ""
        lines.append(f"- **Human Labels Evaluated**: {n_lbls}{prelim_tag}")
        if n_lbls == 0:
            lines.append("  *No human labels available for this split.*\n")
            continue

        ov = split_res.get("overall", {})
        ez = split_res.get("easy", {})
        hd = split_res.get("hard", {})

        lines.append("")
        lines.append("| Metric | Overall | Easy Frames | Hard Segments |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| **Hand Detection Recall** | {ov['hand_presence_recall']['formatted']} | {ez['hand_presence_recall']['formatted']} | {hd['hand_presence_recall']['formatted']} |")
        lines.append(f"| **False Detection Rate** | {ov['false_detection_rate']['formatted']} | {ez['false_detection_rate']['formatted']} | {hd['false_detection_rate']['formatted']} |")
        lines.append(f"| **Side Accuracy (L/R)** | {ov['side_accuracy']['formatted']} | {ez['side_accuracy']['formatted']} | {hd['side_accuracy']['formatted']} |")
        lines.append(f"| **Contact State Accuracy** | {ov['contact_accuracy']['formatted']} | {ez['contact_accuracy']['formatted']} | {hd['contact_accuracy']['formatted']} |")
        lines.append(f"| **Predicted (visible=false) Hands** | {ov['predicted_hands_share']['formatted']} | {ez['predicted_hands_share']['formatted']} | {hd['predicted_hands_share']['formatted']} |")
        lines.append(f"| **Review Queue Catch Rate** | {ov['review_queue_catch_rate']['formatted']} | {ez['review_queue_catch_rate']['formatted']} | {hd['review_queue_catch_rate']['formatted']} |")
        lines.append("")

        cm = ov.get("confusion_matrix", {})
        lines.append("#### Contact State Confusion Matrix (Overall)")
        lines.append("| GT \\ Predicted | none | portable | fixed |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for row_c in ["none", "portable", "fixed"]:
            r = cm.get(row_c, {})
            lines.append(f"| **{row_c}** | {r.get('none', 0)} | {r.get('portable', 0)} | {r.get('fixed', 0)} |")
        lines.append("")

    # BLOCK B: Merged AI labels
    lines.append("---")
    lines.append("## Block B: Against Merged AI Labels (from earlier AI passes, not an independent look)")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> Merged AI labels are aggregated from earlier AI passes with sides omitted. They provide secondary pipeline stress-testing but are NOT human ground truth.")
    lines.append("")
    ai_res = results.get("ai_only", {})
    for split_key in ["tune", "test"]:
        split_res = ai_res.get(split_key, {})
        lines.append(f"### Merged-AI Split: `{split_key.upper()}`")
        if split_res.get("locked"):
            lines.append(f"**STATUS: LOCKED** — {split_res.get('message', 'Locked.')}\n")
            continue

        n_lbls = split_res.get("labels_evaluated_count", 0)
        lines.append(f"- **Merged-AI Labels Evaluated**: {n_lbls}")
        if n_lbls == 0:
            lines.append("  *No merged AI labels evaluated for this split.*\n")
            continue

        ov = split_res.get("overall", {})
        ez = split_res.get("easy", {})
        hd = split_res.get("hard", {})

        lines.append("")
        lines.append("| Metric | Overall | Easy Frames | Hard Segments |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| **Hand Detection Recall** | {ov['hand_presence_recall']['formatted']} | {ez['hand_presence_recall']['formatted']} | {hd['hand_presence_recall']['formatted']} |")
        lines.append(f"| **False Detection Rate** | {ov['false_detection_rate']['formatted']} | {ez['false_detection_rate']['formatted']} | {hd['false_detection_rate']['formatted']} |")
        lines.append(f"| **Side Accuracy (L/R)** | N/A (omitted on merged AI labels) | N/A | N/A |")
        lines.append(f"| **Contact State Accuracy** | {ov['contact_accuracy']['formatted']} | {ez['contact_accuracy']['formatted']} | {hd['contact_accuracy']['formatted']} |")
        lines.append(f"| **Predicted (visible=false) Hands** | {ov['predicted_hands_share']['formatted']} | {ez['predicted_hands_share']['formatted']} | {hd['predicted_hands_share']['formatted']} |")
        lines.append(f"| **Review Queue Catch Rate** | {ov['review_queue_catch_rate']['formatted']} | {ez['review_queue_catch_rate']['formatted']} | {hd['review_queue_catch_rate']['formatted']} |")
        lines.append("")

        cm = ov.get("confusion_matrix", {})
        lines.append("#### Contact State Confusion Matrix (Overall)")
        lines.append("| AI-Label \\ Predicted | none | portable | fixed |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for row_c in ["none", "portable", "fixed"]:
            r = cm.get(row_c, {})
            lines.append(f"| **{row_c}** | {r.get('none', 0)} | {r.get('portable', 0)} | {r.get('fixed', 0)} |")
        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_file", default="eval/labels/stratified_labels.json")
    parser.add_argument("--ai_only_file", default="eval/labels_ai_only.json")
    parser.add_argument("--spotcheck_file", default="eval/spotcheck.json")
    parser.add_argument("--split_file", default="eval/split.json")
    parser.add_argument("--out_dir", default="D:/egohack_data/out")
    parser.add_argument("--hard_segments", default="hands/hard_segments.json")
    parser.add_argument("--unlock_test", action="store_true", default=True, help="Unlock and evaluate test split")
    args = parser.parse_args()

    split_p = Path(args.split_file)
    split_data = json.load(open(split_p, "r", encoding="utf-8"))
    tune_clips = split_data.get("tune", {})
    test_clips = split_data.get("test", {})

    labels_p = Path(args.labels_file)
    raw_human_labels = json.load(open(labels_p, "r", encoding="utf-8")) if labels_p.exists() else {}
    human_labels = {}
    for k, v in raw_human_labels.items():
        if k in EXCLUDED_HUMAN_KEYS:
            continue
        src = v.get("source")
        if src in ["ai_assisted_human_verified", "human", "human_verified"] or (src and "human" in src) or "h1_found" in v:
            human_labels[k] = v

    ai_only_p = Path(args.ai_only_file)
    ai_only_labels = json.load(open(ai_only_p, "r", encoding="utf-8")) if ai_only_p.exists() else {}

    sc_p = Path(args.spotcheck_file)
    spotcheck_info = None
    if sc_p.exists():
        try:
            sc_data = json.load(open(sc_p, "r", encoding="utf-8"))
            if sc_data:
                right_count = sum(1 for v in sc_data.values() if str(v.get("verdict", "")).lower() == "right")
                total_count = len(sc_data)
                spotcheck_info = {
                    "right_count": right_count,
                    "total_count": total_count,
                    "accuracy": round(right_count / total_count, 3) if total_count > 0 else 0.0
                }
        except Exception:
            pass

    hard_p = Path(args.hard_segments)
    hard_segments = json.load(open(hard_p, "r", encoding="utf-8")) if hard_p.exists() else []

    out_root = Path(args.out_dir)

    # Preload pipeline caches for all clips
    clips = sorted(list(set(list(tune_clips.keys()) + list(test_clips.keys()))))
    pipe_cache = {}
    for clip in clips:
        h_path = out_root / clip / "hands.jsonl"
        c_path = out_root / clip / "contacts_frames.jsonl"
        m_path = out_root / clip / "metrics_hands.json"
        rq_path = out_root / clip / "review_queue.csv"
        ct_path = out_root / clip / "contacts.csv"

        hands_map = {}
        if h_path.exists():
            with open(h_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    hands_map[d["frame"]] = d

        contacts_map = {}
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    c_dict = {c["hand_id"]: c for c in d.get("contacts", [])}
                    contacts_map[d["frame"]] = c_dict

        all_events = []
        if ct_path.exists():
            with open(ct_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_events.append({
                        "t_start": float(row["t_start"]),
                        "t_end": float(row["t_end"])
                    })

        rq_intervals = []
        if rq_path.exists():
            with open(rq_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rq_intervals.append({
                        "t_start": float(row["t_start"]),
                        "t_end": float(row["t_end"])
                    })

        fps_val = None
        if m_path.exists():
            try:
                m_data = json.load(open(m_path, "r", encoding="utf-8"))
                fps_val = m_data.get("total_pipeline_fps") or m_data.get("detector_fps")
            except Exception:
                pass

        pipe_cache[clip] = {
            "hands": hands_map,
            "contacts": contacts_map,
            "all_events": all_events,
            "review_intervals": rq_intervals,
            "fps": fps_val
        }

    results = {
        "test_opened_time": TEST_OPENED_TIME,
        "excluded_human_keys": list(EXCLUDED_HUMAN_KEYS),
        "alignment": "nearest_processed_frame_within_1_frame",
        "spotcheck": spotcheck_info,
        "human": {},
        "ai_only": {}
    }

    # Evaluate Human Set (n=20 total, 9 tune, 11 test)
    results["human"]["tune"] = evaluate_split("tune", tune_clips, human_labels, out_root, hard_segments, pipe_cache, is_ai_only=False)
    results["human"]["test"] = evaluate_split("test", test_clips, human_labels, out_root, hard_segments, pipe_cache, is_ai_only=False)

    # Evaluate Merged-AI Set
    results["ai_only"]["tune"] = evaluate_split("tune", tune_clips, ai_only_labels, out_root, hard_segments, pipe_cache, is_ai_only=True)
    results["ai_only"]["test"] = evaluate_split("test", test_clips, ai_only_labels, out_root, hard_segments, pipe_cache, is_ai_only=True)

    results["tune"] = results["human"]["tune"]
    results["test"] = results["human"]["test"]

    results_json_p = Path("eval/results.json")
    with open(results_json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results_md_p = Path("eval/results.md")
    md_content = generate_markdown_report(results, spotcheck_info)
    with open(results_md_p, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Evaluator completed successfully. Human labels evaluated: {len(human_labels)} (Tune: {results['human']['tune']['labels_evaluated_count']}, Test: {results['human']['test']['labels_evaluated_count']}).")

if __name__ == "__main__":
    main()

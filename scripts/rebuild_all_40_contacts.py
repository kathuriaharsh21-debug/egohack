"""Rebuild contacts.csv and review_queue.csv for all 40 clips using the state machine.

State Machine:
- Drop events < 0.3s
- Merge same-hand, same-object, same-state gaps < 0.4s
- Threshold set on tune split to 0.85 (achieving ~23% tune review rate)
- Predicted hand triggered only when vis_share < 0.50 (more than half event frames visible=false)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

IGNORE_CLASSES = {
    "water bottle", "bottle", "phone", "shelf", "box on floor", "gloves", "none", "table"
}

def process_clip_state_machine(
    clip_dir: Path,
    conf_threshold: float = 0.85
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    clip_name = clip_dir.name
    h_path = clip_dir / "hands.jsonl"
    o_path = clip_dir / "objects.jsonl"
    c_path = clip_dir / "contacts_frames.jsonl"

    if not (h_path.exists() and o_path.exists() and c_path.exists()):
        return [], [], 0

    hands_by_frame = {}
    with open(h_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            hands_by_frame[d["frame"]] = d

    objects_by_frame = {}
    with open(o_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            od = {}
            for obj in d.get("objects", []):
                if obj.get("id") is not None:
                    od[obj["id"]] = obj.get("score", 0.0)
                od[obj.get("label", "").lower()] = max(od.get(obj.get("label", "").lower(), 0.0), obj.get("score", 0.0))
            objects_by_frame[d["frame"]] = od

    contact_frames = []
    with open(c_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                contact_frames.append(json.loads(line))

    timeline_by_side = {"left": [], "right": []}
    for fd in contact_frames:
        frame_idx = fd["frame"]
        t = fd["t"]
        for c in fd.get("contacts", []):
            side = c.get("side", "").lower()
            if side in timeline_by_side:
                timeline_by_side[side].append({
                    "frame": frame_idx,
                    "t": t,
                    "hand_id": c.get("hand_id"),
                    "state": c.get("state", "none").lower(),
                    "object_name": c.get("object_name", "none").lower(),
                    "object_id": c.get("object_id"),
                })

    raw_intervals = []
    for side, events in timeline_by_side.items():
        if not events:
            continue
        curr = None
        for ev in events:
            if curr is None:
                curr = {
                    "clip": clip_name,
                    "hand": side,
                    "state": ev["state"],
                    "object_name": ev["object_name"],
                    "object_id": ev["object_id"] if ev["object_id"] is not None else "",
                    "frames": [ev["frame"]],
                    "t_start": ev["t"],
                    "t_end": ev["t"],
                }
            else:
                same_state = (ev["state"] == curr["state"])
                same_obj = (ev["object_name"] == curr["object_name"])
                same_id = (ev["object_id"] == curr["object_id"] or (curr["object_id"] == "" and ev["object_id"] is None))
                if same_state and same_obj and same_id:
                    curr["t_end"] = ev["t"]
                    curr["frames"].append(ev["frame"])
                else:
                    raw_intervals.append(curr)
                    curr = {
                        "clip": clip_name,
                        "hand": side,
                        "state": ev["state"],
                        "object_name": ev["object_name"],
                        "object_id": ev["object_id"] if ev["object_id"] is not None else "",
                        "frames": [ev["frame"]],
                        "t_start": ev["t"],
                        "t_end": ev["t"],
                    }
        if curr is not None:
            raw_intervals.append(curr)

    raw_count = len(raw_intervals)

    # State Machine:
    # 1. Merge same-hand, same-object, same-state gaps < 0.4s
    by_hand = {"left": [], "right": []}
    for inv in raw_intervals:
        by_hand[inv["hand"]].append(inv)

    cleaned_intervals = []
    for side, inv_list in by_hand.items():
        if not inv_list:
            continue
        inv_list = sorted(inv_list, key=lambda x: x["t_start"])
        merged = []
        for inv in inv_list:
            if not merged:
                merged.append(inv)
            else:
                prev = merged[-1]
                gap = inv["t_start"] - prev["t_end"]
                same_state = (inv["state"] == prev["state"])
                same_obj = (inv["object_name"] == prev["object_name"])
                same_id = (inv["object_id"] == prev["object_id"])

                if same_state and same_obj and same_id and 0.0 <= gap < 0.40:
                    prev["t_end"] = inv["t_end"]
                    prev["frames"].extend(inv["frames"])
                else:
                    merged.append(inv)

        # 2. Drop events < 0.3s
        filtered = [inv for inv in merged if (inv["t_end"] - inv["t_start"]) >= 0.30]
        cleaned_intervals.extend(filtered)

    cleaned_intervals = sorted(cleaned_intervals, key=lambda x: (x["t_start"], x["hand"]))

    # Confidence and review flagging
    contacts_rows = []
    review_rows = []

    for inv in cleaned_intervals:
        ev_frames = inv["frames"]
        step_frames = [f for f in ev_frames if f % 2 == 0] or ev_frames

        vis_count = 0
        hand_scores = []
        obj_scores = []
        side_flip_count = 0

        for fr in step_frames:
            h_d = hands_by_frame.get(fr, {})
            hands = h_d.get("hands", [])
            matched = [h for h in hands if h.get("side") == inv["hand"] and h.get("is_wearer")]
            if matched:
                h0 = matched[0]
                if h0.get("visible") and not h0.get("interp"):
                    vis_count += 1
                    hand_scores.append(h0.get("score", 0.0))
            elif hands:
                side_flip_count += 1

            if inv["state"] == "none":
                obj_scores.append(1.0)
            else:
                od = objects_by_frame.get(fr, {})
                s = od.get(inv["object_id"], od.get(inv["object_name"], 0.0))
                obj_scores.append(s)

        vis_share = vis_count / len(step_frames)
        mean_hand = sum(hand_scores) / len(hand_scores) if hand_scores else 0.0
        mean_obj = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0
        side_flip_share = side_flip_count / len(step_frames)

        conf = round(vis_share * mean_hand * mean_obj * (1.0 - side_flip_share), 4)

        contact_entry = {
            "clip": clip_name,
            "t_start": round(inv["t_start"], 3),
            "t_end": round(inv["t_end"], 3),
            "hand": inv["hand"],
            "state": inv["state"],
            "object_name": inv["object_name"],
            "object_id": inv["object_id"],
            "confidence": conf
        }
        contacts_rows.append(contact_entry)

        reasons = []
        if inv["object_name"] in IGNORE_CLASSES and inv["state"] != "none":
            reasons.append("object class in the ignore list")
        if inv["state"] != "none" and mean_obj == 0.0:
            reasons.append("no object score")
        if side_flip_share > 0.0:
            reasons.append("side flip")
        if vis_share < 0.50:  # strictly when more than half of event frames are visible=false
            reasons.append("predicted hand")
        if conf < conf_threshold:
            reasons.append("low confidence")

        if reasons:
            primary_reason = reasons[0]
            review_rows.append({
                "clip": clip_name,
                "t_start": round(inv["t_start"], 3),
                "t_end": round(inv["t_end"], 3),
                "hand": inv["hand"],
                "state": inv["state"],
                "object_name": inv["object_name"],
                "object_id": inv["object_id"],
                "confidence": conf,
                "reason": primary_reason,
                "all_reasons": ";".join(reasons)
            })

    # Write contacts.csv
    contacts_csv_p = clip_dir / "contacts.csv"
    with open(contacts_csv_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip", "t_start", "t_end", "hand", "state", "object_name", "object_id", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contacts_rows)

    # Write review_queue.csv
    review_csv_p = clip_dir / "review_queue.csv"
    with open(review_csv_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip", "t_start", "t_end", "hand", "state", "object_name", "object_id", "confidence", "reason", "all_reasons"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    return contacts_rows, review_rows, raw_count

def main():
    out_dir = Path("D:/egohack_data/out")
    clips = sorted([d.name for d in out_dir.iterdir() if d.is_dir()])
    print(f"Rebuilding contacts and review queues across all {len(clips)} clips with state machine...")

    all_contacts = []
    all_reviews = []
    tot_raw = 0
    dev_clips = ["mytronlabs-000060-006", "MTL-SKL00089-008", "mytronlabs-000003-013", "MTL-SGF00106-036"]
    dev_stats = {}

    for c in clips:
        clip_dir = out_dir / c
        conts, revs, raw_cnt = process_clip_state_machine(clip_dir, conf_threshold=0.85)
        all_contacts.extend(conts)
        all_reviews.extend(revs)
        tot_raw += raw_cnt

        if c in dev_clips:
            dev_stats[c] = {
                "before": raw_cnt,
                "after": len(conts),
                "review": len(revs),
                "share": len(revs) / len(conts) * 100.0 if conts else 0.0
            }

    print("\n=== DEV CLIPS EVENT STATE MACHINE RESULTS ===")
    for dev_c in dev_clips:
        st = dev_stats.get(dev_c, {})
        print(f"[{dev_c}]: Before={st.get('before')} -> After={st.get('after')} events | Review={st.get('review')} ({st.get('share'):.1f}%)")

    # Aggregate review queue
    agg_review_p = Path("eval/review_queue_all.csv")
    with open(agg_review_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip", "t_start", "t_end", "hand", "state", "object_name", "object_id", "confidence", "reason", "all_reasons"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_reviews)

    print(f"\nAll 40 Clips Total: Before={tot_raw} -> After={len(all_contacts)} events")
    print(f"All 40 Clips Total Review: {len(all_reviews)}/{len(all_contacts)} ({len(all_reviews)/len(all_contacts)*100:.1f}%)")

if __name__ == "__main__":
    main()


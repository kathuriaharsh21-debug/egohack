"""Generates contacts.csv from contacts_frames.jsonl.

Groups frame-by-frame contact states into continuous temporal intervals:
Columns: clip,t_start,t_end,hand,state,object_name,object_id,confidence
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def generate_contacts_csv(
    clip_name: str,
    contacts_frames_path: Path,
    output_csv_path: Path,
) -> List[Dict[str, Any]]:
    """Convert per-frame contacts into interval segments for contacts.csv."""
    if not contacts_frames_path.exists():
        return []

    # Read per-frame contacts
    frames_data = []
    with open(contacts_frames_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames_data.append(json.loads(line))

    # Separate timelines by hand side: 'left' and 'right'
    timeline_by_side: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": []}
    for fd in frames_data:
        t = fd["t"]
        for c in fd.get("contacts", []):
            side = c.get("side", "").lower()
            if side in timeline_by_side:
                timeline_by_side[side].append({
                    "t": t,
                    "state": c.get("state", "none"),
                    "object_name": c.get("object_name", "none"),
                    "object_id": c.get("object_id"),
                    "confidence": c.get("confidence", 0.0),
                })

    intervals: List[Dict[str, Any]] = []

    for side, events in timeline_by_side.items():
        if not events:
            continue

        curr_interval = None
        for ev in events:
            if curr_interval is None:
                curr_interval = {
                    "clip": clip_name,
                    "t_start": ev["t"],
                    "t_end": ev["t"],
                    "hand": side,
                    "state": ev["state"],
                    "object_name": ev["object_name"],
                    "object_id": ev["object_id"] if ev["object_id"] is not None else "",
                    "confs": [ev["confidence"]],
                }
            else:
                # Check if state and object match
                same_state = (ev["state"] == curr_interval["state"])
                same_obj = (ev["object_name"] == curr_interval["object_name"])
                same_id = (ev["object_id"] == curr_interval["object_id"] or (curr_interval["object_id"] == "" and ev["object_id"] is None))

                if same_state and same_obj and same_id:
                    curr_interval["t_end"] = ev["t"]
                    curr_interval["confs"].append(ev["confidence"])
                else:
                    # Finalize current interval
                    avg_conf = sum(curr_interval["confs"]) / len(curr_interval["confs"])
                    intervals.append({
                        "clip": curr_interval["clip"],
                        "t_start": round(curr_interval["t_start"], 3),
                        "t_end": round(curr_interval["t_end"], 3),
                        "hand": curr_interval["hand"],
                        "state": curr_interval["state"],
                        "object_name": curr_interval["object_name"],
                        "object_id": curr_interval["object_id"],
                        "confidence": round(avg_conf, 4),
                    })
                    curr_interval = {
                        "clip": clip_name,
                        "t_start": ev["t"],
                        "t_end": ev["t"],
                        "hand": side,
                        "state": ev["state"],
                        "object_name": ev["object_name"],
                        "object_id": ev["object_id"] if ev["object_id"] is not None else "",
                        "confs": [ev["confidence"]],
                    }

        if curr_interval is not None:
            avg_conf = sum(curr_interval["confs"]) / len(curr_interval["confs"])
            intervals.append({
                "clip": curr_interval["clip"],
                "t_start": round(curr_interval["t_start"], 3),
                "t_end": round(curr_interval["t_end"], 3),
                "hand": curr_interval["hand"],
                "state": curr_interval["state"],
                "object_name": curr_interval["object_name"],
                "object_id": curr_interval["object_id"],
                "confidence": round(avg_conf, 4),
            })

    # Sort intervals by t_start, then hand
    intervals.sort(key=lambda x: (x["t_start"], x["hand"]))

    # Write contacts.csv
    fieldnames = ["clip", "t_start", "t_end", "hand", "state", "object_name", "object_id", "confidence"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in intervals:
            writer.writerow(row)

    return intervals

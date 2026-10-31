"""Streamlit Human Labeling and Evaluation Tool for EgoHack.

Supports:
1. Frame-by-frame annotation on 100 stratified frames (wearer vs bystander, side correctness,
   21 points pass/fail, contact state: none/portable/fixed, object name).
2. Video timeline scrubbing mode for marking grab / release contact events.
Saves labels to eval/labels/ (gitignored).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

LABELS_DIR = Path("eval/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DEV_CLIPS = [
    "mytronlabs-000060-006",
    "MTL-SKL00089-008",
    "mytronlabs-000003-013",
    "MTL-SGF00106-036",
]


@st.cache_data
def get_stratified_frames(total_samples: int = 100, seed: int = 42) -> List[dict]:
    """Sample 100 frames stratified across dev clips and easy/hard segments."""
    np.random.seed(seed)
    samples = []
    samples_per_clip = total_samples // len(DEV_CLIPS)

    for clip_name in DEV_CLIPS:
        clip_path = Path("data/clips_480p") / f"{clip_name}.mp4"
        if not clip_path.exists():
            continue
        cap = cv2.VideoCapture(str(clip_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Stratify: first 20s (0-600 frames) vs remaining (600-total_f)
        early_frames = np.random.choice(min(600, total_f), size=samples_per_clip // 2, replace=False)
        late_frames = np.random.choice(range(min(600, total_f), total_f), size=samples_per_clip - len(early_frames), replace=False)
        clip_frames = sorted(list(early_frames) + list(late_frames))

        for f_idx in clip_frames:
            samples.append({
                "clip": clip_name,
                "frame": int(f_idx),
                "t": round(f_idx / fps, 3),
            })

    return samples


def load_frame_image(clip_name: str, frame_idx: int) -> np.ndarray:
    clip_path = Path("data/clips_480p") / f"{clip_name}.mp4"
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((480, 854, 3), dtype=np.uint8)


def load_hand_predictions(clip_name: str, frame_idx: int) -> list:
    jsonl_path = Path("out") / clip_name / "hands.jsonl"
    if not jsonl_path.exists():
        return []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("frame") == frame_idx:
                return data.get("hands", [])
    return []


def main():
    st.set_page_config(page_title="EgoHack Labeling Tool", layout="wide")
    st.title("🎯 EgoHack Auto-Annotation & Ground Truth Tool")

    mode = st.sidebar.radio("Select Tool Mode", ["Frame Annotation (100 Stratified)", "Timeline Contact Scrubber"])

    if mode == "Frame Annotation (100 Stratified)":
        samples = get_stratified_frames(100, seed=42)
        sample_idx = st.sidebar.slider("Sample Index", 0, len(samples) - 1, 0)
        curr = samples[sample_idx]

        st.subheader(f"Clip: `{curr['clip']}` | Frame: `{curr['frame']}` ({curr['t']}s)")

        col_img, col_form = st.columns([1.4, 1.0])

        img = load_frame_image(curr['clip'], curr['frame'])
        preds = load_hand_predictions(curr['clip'], curr['frame'])

        # Draw predictions if available
        disp_img = img.copy()
        for h in preds:
            bx = [int(v) for v in h.get("bbox", [0, 0, 0, 0])]
            color = (0, 255, 0) if h.get("side") == "left" else (255, 140, 0)
            cv2.rectangle(disp_img, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(disp_img, f"{h.get('side')} ({h.get('score', 0):.2f})", (bx[0], max(15, bx[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with col_img:
            st.image(disp_img, use_container_width=True, caption=f"Frame {curr['frame']} with pipeline detection overlay")

        # Load existing labels
        labels_file = LABELS_DIR / "stratified_labels.json"
        existing_labels = {}
        if labels_file.exists():
            with open(labels_file, "r") as f:
                existing_labels = json.load(f)

        key = f"{curr['clip']}_{curr['frame']}"
        prev = existing_labels.get(key, {})

        with col_form:
            st.markdown("### Hand 1 (e.g. Left / Primary)")
            h1_found = st.selectbox("Hand 1 Found?", ["Yes", "No"], index=0 if prev.get("h1_found", "Yes") == "Yes" else 1, key="h1_found")
            h1_role = st.selectbox("Hand 1 Role", ["Wearer", "Other Person (Bystander)"], index=0 if prev.get("h1_role") != "Other Person (Bystander)" else 1, key="h1_role")
            h1_side = st.selectbox("Hand 1 True Side", ["left", "right"], index=0 if prev.get("h1_side", "left") == "left" else 1, key="h1_side")
            h1_kp_pass = st.selectbox("21 Keypoints Quality", ["Pass", "Fail"], index=0 if prev.get("h1_kp_pass", "Pass") == "Pass" else 1, key="h1_kp")
            h1_contact = st.selectbox("Contact State", ["none", "portable", "fixed"], index=["none", "portable", "fixed"].index(prev.get("h1_contact", "none")), key="h1_contact")
            h1_obj = st.text_input("Object Name", value=prev.get("h1_obj", ""), key="h1_obj")

            st.markdown("---")
            st.markdown("### Hand 2 (e.g. Right / Secondary)")
            h2_present = st.checkbox("Hand 2 is present in frame", value=prev.get("h2_present", False), key="h2_pres")
            h2_data = {}
            if h2_present:
                h2_found = st.selectbox("Hand 2 Found?", ["Yes", "No"], key="h2_found")
                h2_role = st.selectbox("Hand 2 Role", ["Wearer", "Other Person (Bystander)"], key="h2_role")
                h2_side = st.selectbox("Hand 2 True Side", ["right", "left"], key="h2_side")
                h2_kp_pass = st.selectbox("Hand 2 Keypoints", ["Pass", "Fail"], key="h2_kp")
                h2_contact = st.selectbox("Hand 2 Contact State", ["none", "portable", "fixed"], key="h2_contact")
                h2_obj = st.text_input("Hand 2 Object Name", value="", key="h2_obj")
                h2_data = {
                    "h2_present": True, "h2_found": h2_found, "h2_role": h2_role,
                    "h2_side": h2_side, "h2_kp_pass": h2_kp_pass, "h2_contact": h2_contact, "h2_obj": h2_obj
                }

            if st.button("💾 Save Annotation", type="primary"):
                existing_labels[key] = {
                    "clip": curr["clip"],
                    "frame": curr["frame"],
                    "t": curr["t"],
                    "h1_found": h1_found,
                    "h1_role": h1_role,
                    "h1_side": h1_side,
                    "h1_kp_pass": h1_kp_pass,
                    "h1_contact": h1_contact,
                    "h1_obj": h1_obj,
                    **h2_data
                }
                with open(labels_file, "w") as f:
                    json.dump(existing_labels, f, indent=2)
                st.success(f"Saved annotation for frame {curr['frame']}! ({len(existing_labels)} total labeled)")

    elif mode == "Timeline Contact Scrubber":
        clip = st.sidebar.selectbox("Choose Clip", DEV_CLIPS)
        clip_path = Path("data/clips_480p") / f"{clip}.mp4"
        cap = cv2.VideoCapture(str(clip_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        t_slider = st.slider("Time Scrub (seconds)", 0.0, float(total_f / fps), 0.0, step=0.1)
        curr_f = int(round(t_slider * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption=f"{clip} @ {t_slider:.2f}s (Frame {curr_f})")

        st.markdown("### ⏱️ Mark Interaction Event")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            t_start = st.number_input("t_start (s)", value=float(t_slider), step=0.1)
        with c2:
            t_end = st.number_input("t_end (s)", value=float(t_slider + 1.0), step=0.1)
        with c3:
            hand_side = st.selectbox("Hand", ["left", "right"])
            state = st.selectbox("Contact State", ["none", "portable", "fixed"])
        with c4:
            obj_name = st.text_input("Object Name (e.g. tape dispenser, saw, roller)")

        contacts_csv = LABELS_DIR / "contacts_ground_truth.csv"
        if st.button("➕ Add Contact Event"):
            new_row = {
                "clip": clip,
                "t_start": t_start,
                "t_end": t_end,
                "hand": hand_side,
                "state": state,
                "object_name": obj_name,
            }
            if contacts_csv.exists():
                df = pd.read_csv(contacts_csv)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(contacts_csv, index=False)
            st.success(f"Added event: {hand_side} touching {obj_name} ({state}) from {t_start}s to {t_end}s")

        if contacts_csv.exists():
            st.markdown("#### Existing Contact Ground Truth")
            st.dataframe(pd.read_csv(contacts_csv))


if __name__ == "__main__":
    main()

"""Streamlit Human Labeling and Evaluation Tool for EgoHack.

Supports:
1. Review AI Labels mode: inspect AI pre-labels (side hidden, hints only, user decides side).
   Fix mode starts empty with "-- choose --" and all fields required before saving.
2. Frame Annotation (Manual): all fields start at "-- choose --" and are required.
   Role, side, keypoints, and contact greyed out when Found = No or NO_HAND.
3. Video timeline scrubbing mode for marking grab / release contact events.
Saves labels to eval/labels/ (gitignored).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

LABELS_DIR = Path("eval/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)
AI_LABELS_DIR = Path("eval/labels_ai")
AI_FRAMES_DIR = Path("eval/ai_frames")


def save_labels_atomic(labels_path: Path, data: dict):
    """Atomically write data to labels_path using a temporary file and os.replace."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = labels_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, labels_path)


def get_option_idx(val: Any, options: list, default_idx: int = 0) -> int:
    """Return index of val in options if present, else default_idx."""
    if val in options:
        return options.index(val)
    return default_idx

DEV_CLIPS = [
    "mytronlabs-000060-006",
    "MTL-SKL00089-008",
    "mytronlabs-000003-013",
    "MTL-SGF00106-036",
]


def get_clip_480p_path(clip_name: str) -> Path:
    p1 = Path("C:/egohack_data/clips_480p") / f"{clip_name}.mp4"
    if p1.exists():
        return p1
    return Path("data/clips_480p") / f"{clip_name}.mp4"


@st.cache_data
def get_stratified_frames(total_samples: int = 100, seed: int = 42) -> List[dict]:
    """Sample 100 frames stratified across dev clips and easy/hard segments."""
    np.random.seed(seed)
    samples = []
    samples_per_clip = total_samples // len(DEV_CLIPS)

    for clip_name in DEV_CLIPS:
        clip_path = get_clip_480p_path(clip_name)
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
    """Load frame from eval/ai_frames if cached, else from MP4."""
    cached_frame = AI_FRAMES_DIR / f"{clip_name}_{frame_idx}.png"
    if cached_frame.exists():
        img = cv2.imread(str(cached_frame))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clip_path = get_clip_480p_path(clip_name)
    if not clip_path.exists():
        return np.zeros((480, 854, 3), dtype=np.uint8)
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


def load_ai_label(clip_name: str, frame_idx: int) -> Optional[dict]:
    ai_file = AI_LABELS_DIR / f"{clip_name}_{frame_idx}.json"
    if ai_file.exists():
        with open(ai_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_ai_review_samples() -> List[dict]:
    """Return samples ordered by eval/review_queue_100.csv (disagreements, unsure, then agreed)."""
    queue_path = Path("eval/review_queue_100.csv")
    if not queue_path.exists():
        return []

    df_q = pd.read_csv(queue_path)
    order_map = {"a_disagreement": 0, "b_unsure": 1, "c_agreed": 2}
    if "priority_group" in df_q.columns:
        df_q["_sort_key"] = df_q["priority_group"].map(lambda x: order_map.get(str(x), 99))
        df_q = df_q.sort_values(by=["_sort_key"]).drop(columns=["_sort_key"])

    samples = []
    for _, row in df_q.iterrows():
        clip_name = row["clip"]
        frame_num = int(row["frame"])
        ai_data = load_ai_label(clip_name, frame_num) or {}
        grp = str(row.get("priority_group", ""))
        default_prio = 0 if "disagreement" in grp else (1 if "unsure" in grp else 2)
        prio = int(row.get("priority", default_prio))
        samples.append({
            "clip": clip_name,
            "frame": frame_num,
            "t": float(row.get("t", 0.0)),
            "priority": prio,
            "priority_group": grp,
            "disagreements": str(row.get("disagreements", "")),
            "ai_data": ai_data
        })
    return samples


def main():
    st.set_page_config(page_title="EgoHack Labeling & Review Tool", layout="wide")
    st.title("🎯 EgoHack Auto-Annotation & Ground Truth Tool")

    mode = st.sidebar.radio(
        "Select Tool Mode",
        [
            "Review AI Labels (Phase 2)",
            "Frame Annotation (Manual)",
            "Timeline Contact Scrubber"
        ]
    )

    labels_file = LABELS_DIR / "stratified_labels.json"
    existing_labels = {}
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            try:
                existing_labels = json.load(f)
            except Exception:
                existing_labels = {}

    if mode == "Review AI Labels (Phase 2)":
        queue_path = Path("eval/review_queue_100.csv")
        if not queue_path.exists():
            st.error("⚠️ Review queue file `eval/review_queue_100.csv` is missing. Please generate it before running Review mode.")
            return

        samples = get_ai_review_samples()
        total_samples = len(samples)
        if total_samples == 0:
            st.warning("⚠️ Review queue `eval/review_queue_100.csv` is empty.")
            return

        verified_count = sum(1 for s in samples if f"{s['clip']}_{s['frame']}" in existing_labels)

        st.sidebar.markdown(f"### Review Progress: **{verified_count} / {total_samples}**")
        st.sidebar.progress(verified_count / max(1, total_samples))
        st.sidebar.caption("Ordering: ⚠️ Disagreements first, then Unsure frames, then Agreed.")

        sample_idx = st.sidebar.slider("Select Sample to Review", 0, max(0, total_samples - 1), 0)
        curr = samples[min(sample_idx, total_samples - 1)]
        key = f"{curr['clip']}_{curr['frame']}"
        ai_data = curr.get("ai_data", {})
        prev = existing_labels.get(key, {})

        col_title, col_status = st.columns([2.5, 1.5])
        with col_title:
            st.subheader(f"Clip: `{curr['clip']}` | Frame: `{curr['frame']}` ({curr['t']}s)")
        with col_status:
            prio = curr.get("priority", 0)
            if prio == 0:
                st.error("⚠️ HIGH REVIEW PRIORITY (Disagreements)")
            elif prio == 1:
                st.warning("⚡ MEDIUM CONFIDENCE (Unsure)")
            else:
                st.success("✅ HIGH CONFIDENCE (Agreed)")

            # Badge: Saved vs Not saved yet
            if key in existing_labels:
                st.success(f"🟢 **SAVED** (Source: `{prev.get('source', 'human_verified')}`)")
            else:
                st.warning("⚪ **NOT SAVED YET**")

        col_img, col_review = st.columns([1.3, 1.1])

        with col_img:
            img = load_frame_image(curr['clip'], curr['frame'])
            show_overlay = st.checkbox("Overlay model detection bounding boxes", value=False, key=f"ov_{key}")
            if show_overlay:
                disp_img = img.copy()
                preds = load_hand_predictions(curr['clip'], curr['frame'])
                for h in preds:
                    bx = [int(v) for v in h.get("bbox", [0, 0, 0, 0])]
                    color = (0, 255, 0) if h.get("side") == "left" else (255, 140, 0)
                    cv2.rectangle(disp_img, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                    cv2.putText(disp_img, f"{h.get('side')} ({h.get('score', 0):.2f})",
                                (bx[0], max(15, bx[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                st.image(disp_img, use_container_width=True, caption=f"Frame {curr['frame']} with pipeline detection overlay")
            else:
                st.image(img, use_container_width=True, caption=f"Raw Frame {curr['frame']} (No overlay)")

        with col_review:
            if "last_save_msg" in st.session_state:
                st.info(f"✅ {st.session_state['last_save_msg']}")

            # 1. HIDE AI side field completely. Show other fields as hints only.
            st.markdown("### 💡 AI Hints (for reference only)")
            st.markdown(
                f"""
                - **Hand Visible**: `{ai_data.get('hand_visible', 'unsure')}`
                - **Role**: `{ai_data.get('role', 'unsure')}`
                - **Contact State**: `{ai_data.get('contact', 'unsure')}`
                - **Object Name**: `{ai_data.get('object_name', '')}`
                - **Hand 2 Visible**: `{ai_data.get('hand2_visible', 'no')}`
                - **Confidence**: `{ai_data.get('confidence', 'unsure')}`
                - **AI Reason**: *"{ai_data.get('reason', '')}"*
                *(Note: AI side prediction is hidden. You must decide side independently.)*
                """
            )

            st.markdown("---")
            st.markdown("#### ✍️ Human Annotation (from scratch)")

            FOUND_OPTIONS = ["-- choose --", "Yes (skeleton on a real hand)", "Missed (real hand, no skeleton)", "No hand"]
            ROLE_OPTIONS = ["-- choose --", "Wearer", "Other Person (Bystander)"]
            SIDE_OPTIONS = ["-- choose --", "left", "right", "unsure"]
            KP_OPTIONS = ["-- choose --", "Pass", "Fail"]
            CONTACT_OPTIONS = ["-- choose --", "none", "portable", "fixed"]

            # Pre-load previous values if saved
            h1_found_idx = get_option_idx(prev.get("h1_found"), FOUND_OPTIONS, 0)
            f_h1_found = st.selectbox("Hand 1 Found?*", FOUND_OPTIONS, index=h1_found_idx, key=f"f_h1_found_{key}")
            f_is_no = (f_h1_found == "No hand")

            h1_role_idx = get_option_idx(prev.get("h1_role"), ROLE_OPTIONS, 0)
            f_h1_role = st.selectbox("Hand 1 Role*", ROLE_OPTIONS, index=h1_role_idx, disabled=f_is_no, key=f"f_h1_role_{key}")

            h1_side_idx = get_option_idx(prev.get("h1_side"), SIDE_OPTIONS, 0)
            f_h1_side = st.selectbox("Hand 1 Side*", SIDE_OPTIONS, index=h1_side_idx, disabled=f_is_no, key=f"f_h1_side_{key}")

            h1_kp_idx = get_option_idx(prev.get("h1_kp_pass"), KP_OPTIONS, 0)
            f_h1_kp = st.selectbox("Keypoints Quality*", KP_OPTIONS, index=h1_kp_idx, disabled=f_is_no, key=f"f_h1_kp_{key}")

            h1_contact_idx = get_option_idx(prev.get("h1_contact"), CONTACT_OPTIONS, 0)
            f_h1_contact = st.selectbox("Contact State*", CONTACT_OPTIONS, index=h1_contact_idx, disabled=f_is_no, key=f"f_h1_contact_{key}")

            prev_h1_obj = prev.get("h1_obj", "")
            f_h1_obj = st.text_input("Object Name* (or NO_HAND)", value="NO_HAND" if f_is_no else prev_h1_obj, disabled=f_is_no, key=f"f_h1_obj_{key}")

            st.markdown("##### Hand 2")
            h2_found_idx = get_option_idx(prev.get("h2_found"), FOUND_OPTIONS, 0)
            f_h2_found = st.selectbox("Hand 2 Found?*", FOUND_OPTIONS, index=h2_found_idx, key=f"f_h2_found_{key}")
            f_h2_is_no = (f_h2_found == "No hand")

            h2_role_idx = get_option_idx(prev.get("h2_role"), ROLE_OPTIONS, 0)
            f_h2_role = st.selectbox("Hand 2 Role*", ROLE_OPTIONS, index=h2_role_idx, disabled=f_h2_is_no, key=f"f_h2_role_{key}")

            h2_side_idx = get_option_idx(prev.get("h2_side"), ["-- choose --", "right", "left", "unsure"], 0)
            f_h2_side = st.selectbox("Hand 2 Side*", ["-- choose --", "right", "left", "unsure"], index=h2_side_idx, disabled=f_h2_is_no, key=f"f_h2_side_{key}")

            h2_kp_idx = get_option_idx(prev.get("h2_kp_pass"), KP_OPTIONS, 0)
            f_h2_kp = st.selectbox("Hand 2 Keypoints*", KP_OPTIONS, index=h2_kp_idx, disabled=f_h2_is_no, key=f"f_h2_kp_{key}")

            h2_contact_idx = get_option_idx(prev.get("h2_contact"), CONTACT_OPTIONS, 0)
            f_h2_contact = st.selectbox("Hand 2 Contact State*", CONTACT_OPTIONS, index=h2_contact_idx, disabled=f_h2_is_no, key=f"f_h2_contact_{key}")

            prev_h2_obj = prev.get("h2_obj", "")
            f_h2_obj = st.text_input("Hand 2 Object Name*", value="NO_HAND" if f_h2_is_no else prev_h2_obj, disabled=f_h2_is_no, key=f"f_h2_obj_{key}")

            if st.button("💾 Save Human Label", type="primary", use_container_width=True, key=f"btn_save_{key}"):
                if f_h1_found == "-- choose --" or f_h2_found == "-- choose --":
                    st.error("❌ Both Hand 1 Found and Hand 2 Found questions must be answered.")
                elif not f_is_no and (f_h1_role == "-- choose --" or f_h1_side == "-- choose --" or f_h1_kp == "-- choose --" or f_h1_contact == "-- choose --" or not f_h1_obj.strip()):
                    st.error("❌ All Hand 1 fields (Role, Side, Keypoints, Contact, Object) are required before saving.")
                elif not f_h2_is_no and (f_h2_role == "-- choose --" or f_h2_side == "-- choose --" or f_h2_kp == "-- choose --" or f_h2_contact == "-- choose --" or not f_h2_obj.strip()):
                    st.error("❌ All Hand 2 fields (Role, Side, Keypoints, Contact, Object) are required when Hand 2 is marked found/missed.")
                else:
                    existing_labels[key] = {
                        "clip": curr["clip"],
                        "frame": curr["frame"],
                        "t": curr["t"],
                        "h1_found": f_h1_found,
                        "h1_role": "None" if f_is_no else f_h1_role,
                        "h1_side": "None" if f_is_no else f_h1_side,
                        "h1_kp_pass": "Fail" if f_is_no else f_h1_kp,
                        "h1_contact": "none" if f_is_no else f_h1_contact,
                        "h1_obj": "NO_HAND" if f_is_no else f_h1_obj.strip(),
                        "h2_found": f_h2_found,
                        "h2_present": not f_h2_is_no,
                        "h2_role": "None" if f_h2_is_no else f_h2_role,
                        "h2_side": "None" if f_h2_is_no else f_h2_side,
                        "h2_kp_pass": "Fail" if f_h2_is_no else f_h2_kp,
                        "h2_contact": "none" if f_h2_is_no else f_h2_contact,
                        "h2_obj": "NO_HAND" if f_h2_is_no else f_h2_obj.strip(),
                        "source": "human_verified",
                        "queue_group": curr.get("priority_group", "")
                    }
                    save_labels_atomic(labels_file, existing_labels)
                    save_time = datetime.now().strftime("%H:%M:%S")
                    s_h1_side = "None" if f_is_no else f_h1_side
                    s_h1_c = "none" if f_is_no else f_h1_contact
                    s_h1_o = "NO_HAND" if f_is_no else f_h1_obj.strip()
                    s_h2_side = "None" if f_h2_is_no else f_h2_side
                    s_h2_c = "none" if f_h2_is_no else f_h2_contact
                    s_h2_o = "NO_HAND" if f_h2_is_no else f_h2_obj.strip()
                    msg = f"Saved at {save_time}: {curr['clip']} f{curr['frame']} | H1: [{f_h1_found}, {s_h1_side}, {s_h1_c}, {s_h1_o}] | H2: [{f_h2_found}, {s_h2_side}, {s_h2_c}, {s_h2_o}]"
                    st.session_state["last_save_msg"] = msg
                    st.success(f"✅ {msg}")
                    st.rerun()

        # Last saved labels table (last 5)
        if existing_labels:
            st.markdown("---")
            st.markdown("### 📋 Last Saved Labels (Most Recent 5)")
            last_keys = list(existing_labels.keys())[-5:]
            recent_rows = []
            for lk in reversed(last_keys):
                item = existing_labels[lk]
                recent_rows.append({
                    "clip": item.get("clip"),
                    "frame": item.get("frame"),
                    "t": item.get("t"),
                    "h1_found": item.get("h1_found"),
                    "h1_side": item.get("h1_side"),
                    "h1_contact": item.get("h1_contact"),
                    "h1_obj": item.get("h1_obj"),
                    "h2_found": item.get("h2_found"),
                    "h2_side": item.get("h2_side"),
                    "source": item.get("source", "unknown")
                })
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True)

    # 3. Normal form: start at "-- choose --", required, grey out when Found = No or NO_HAND.
    elif mode == "Frame Annotation (Manual)":
        samples = get_stratified_frames(100, seed=42)
        sample_idx = st.sidebar.slider("Sample Index", 0, len(samples) - 1, 0)
        curr = samples[sample_idx]

        st.subheader(f"Clip: `{curr['clip']}` | Frame: `{curr['frame']}` ({curr['t']}s)")

        col_img, col_form = st.columns([1.4, 1.0])

        img = load_frame_image(curr['clip'], curr['frame'])
        preds = load_hand_predictions(curr['clip'], curr['frame'])

        disp_img = img.copy()
        for h in preds:
            bx = [int(v) for v in h.get("bbox", [0, 0, 0, 0])]
            color = (0, 255, 0) if h.get("side") == "left" else (255, 140, 0)
            cv2.rectangle(disp_img, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(disp_img, f"{h.get('side')} ({h.get('score', 0):.2f})", (bx[0], max(15, bx[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with col_img:
            st.image(disp_img, use_container_width=True, caption=f"Frame {curr['frame']} with pipeline detection overlay")

        key = f"{curr['clip']}_{curr['frame']}"
        prev = existing_labels.get(key, {})

        if key in existing_labels:
            st.success(f"🟢 **SAVED** (Source: `{prev.get('source', 'manual')}`)")
        else:
            st.warning("⚪ **NOT SAVED YET**")

        with col_form:
            if "last_save_msg" in st.session_state:
                st.info(f"✅ {st.session_state['last_save_msg']}")

            FOUND_OPTIONS = ["-- choose --", "Yes (skeleton on a real hand)", "Missed (real hand, no skeleton)", "No hand"]
            ROLE_OPTIONS = ["-- choose --", "Wearer", "Other Person (Bystander)"]
            SIDE_OPTIONS = ["-- choose --", "left", "right", "unsure"]
            KP_OPTIONS = ["-- choose --", "Pass", "Fail"]
            CONTACT_OPTIONS = ["-- choose --", "none", "portable", "fixed"]

            st.markdown("### Hand 1 (e.g. Left / Primary)")
            h1_found_idx = get_option_idx(prev.get("h1_found"), FOUND_OPTIONS, 0)
            h1_found = st.selectbox("Hand 1 Found?*", FOUND_OPTIONS, index=h1_found_idx, key=f"m_h1_found_{key}")
            is_no_or_nohand = (h1_found == "No hand")

            h1_role_idx = get_option_idx(prev.get("h1_role"), ROLE_OPTIONS, 0)
            h1_role = st.selectbox("Hand 1 Role*", ROLE_OPTIONS, index=h1_role_idx, disabled=is_no_or_nohand, key=f"m_h1_role_{key}")

            h1_side_idx = get_option_idx(prev.get("h1_side"), SIDE_OPTIONS, 0)
            h1_side = st.selectbox("Hand 1 True Side*", SIDE_OPTIONS, index=h1_side_idx, disabled=is_no_or_nohand, key=f"m_h1_side_{key}")

            h1_kp_idx = get_option_idx(prev.get("h1_kp_pass"), KP_OPTIONS, 0)
            h1_kp_pass = st.selectbox("21 Keypoints Quality*", KP_OPTIONS, index=h1_kp_idx, disabled=is_no_or_nohand, key=f"m_h1_kp_{key}")

            h1_contact_idx = get_option_idx(prev.get("h1_contact"), CONTACT_OPTIONS, 0)
            h1_contact = st.selectbox("Contact State*", CONTACT_OPTIONS, index=h1_contact_idx, disabled=is_no_or_nohand, key=f"m_h1_contact_{key}")

            prev_m_h1_obj = prev.get("h1_obj", "")
            h1_obj = st.text_input("Object Name* (or NO_HAND)", value="NO_HAND" if is_no_or_nohand else prev_m_h1_obj, disabled=is_no_or_nohand, key=f"m_h1_obj_{key}")

            st.markdown("---")
            st.markdown("### Hand 2 (e.g. Right / Secondary)")
            h2_found_idx = get_option_idx(prev.get("h2_found"), FOUND_OPTIONS, 0)
            h2_found = st.selectbox("Hand 2 Found?*", FOUND_OPTIONS, index=h2_found_idx, key=f"m_h2_found_{key}")
            h2_is_no = (h2_found == "No hand")

            h2_role_idx = get_option_idx(prev.get("h2_role"), ROLE_OPTIONS, 0)
            h2_role = st.selectbox("Hand 2 Role*", ROLE_OPTIONS, index=h2_role_idx, disabled=h2_is_no, key=f"m_h2_role_{key}")

            h2_side_idx = get_option_idx(prev.get("h2_side"), ["-- choose --", "right", "left", "unsure"], 0)
            h2_side = st.selectbox("Hand 2 True Side*", ["-- choose --", "right", "left", "unsure"], index=h2_side_idx, disabled=h2_is_no, key=f"m_h2_side_{key}")

            h2_kp_idx = get_option_idx(prev.get("h2_kp_pass"), KP_OPTIONS, 0)
            h2_kp_pass = st.selectbox("Hand 2 Keypoints*", KP_OPTIONS, index=h2_kp_idx, disabled=h2_is_no, key=f"m_h2_kp_{key}")

            h2_contact_idx = get_option_idx(prev.get("h2_contact"), CONTACT_OPTIONS, 0)
            h2_contact = st.selectbox("Hand 2 Contact State*", CONTACT_OPTIONS, index=h2_contact_idx, disabled=h2_is_no, key=f"m_h2_contact_{key}")

            prev_m_h2_obj = prev.get("h2_obj", "")
            h2_obj = st.text_input("Hand 2 Object Name*", value="NO_HAND" if h2_is_no else prev_m_h2_obj, disabled=h2_is_no, key=f"m_h2_obj_{key}")

            if st.button("💾 Save Annotation", type="primary", key=f"m_save_btn_{key}"):
                if h1_found == "-- choose --" or h2_found == "-- choose --":
                    st.error("❌ Both Hand 1 Found and Hand 2 Found questions must be answered.")
                elif not is_no_or_nohand and (h1_role == "-- choose --" or h1_side == "-- choose --" or h1_kp_pass == "-- choose --" or h1_contact == "-- choose --" or not h1_obj.strip()):
                    st.error("❌ All Hand 1 fields (Role, Side, Keypoints, Contact, Object) are required before saving.")
                elif not h2_is_no and (h2_role == "-- choose --" or h2_side == "-- choose --" or h2_kp_pass == "-- choose --" or h2_contact == "-- choose --" or not h2_obj.strip()):
                    st.error("❌ All Hand 2 fields (Role, Side, Keypoints, Contact, Object) are required when Hand 2 is marked found/missed.")
                else:
                    existing_labels[key] = {
                        "clip": curr["clip"],
                        "frame": curr["frame"],
                        "t": curr["t"],
                        "h1_found": h1_found,
                        "h1_role": "None" if is_no_or_nohand else h1_role,
                        "h1_side": "None" if is_no_or_nohand else h1_side,
                        "h1_kp_pass": "Fail" if is_no_or_nohand else h1_kp_pass,
                        "h1_contact": "none" if is_no_or_nohand else h1_contact,
                        "h1_obj": "NO_HAND" if is_no_or_nohand else h1_obj.strip(),
                        "h2_found": h2_found,
                        "h2_present": not h2_is_no,
                        "h2_role": "None" if h2_is_no else h2_role,
                        "h2_side": "None" if h2_is_no else h2_side,
                        "h2_kp_pass": "Fail" if h2_is_no else h2_kp_pass,
                        "h2_contact": "none" if h2_is_no else h2_contact,
                        "h2_obj": "NO_HAND" if h2_is_no else h2_obj.strip(),
                        "source": "manual"
                    }
                    save_labels_atomic(labels_file, existing_labels)
                    save_time = datetime.now().strftime("%H:%M:%S")
                    s_h1_side = "None" if is_no_or_nohand else h1_side
                    s_h1_c = "none" if is_no_or_nohand else h1_contact
                    s_h1_o = "NO_HAND" if is_no_or_nohand else h1_obj.strip()
                    s_h2_side = "None" if h2_is_no else h2_side
                    s_h2_c = "none" if h2_is_no else h2_contact
                    s_h2_o = "NO_HAND" if h2_is_no else h2_obj.strip()
                    msg = f"Saved at {save_time}: {curr['clip']} f{curr['frame']} | H1: [{h1_found}, {s_h1_side}, {s_h1_c}, {s_h1_o}] | H2: [{h2_found}, {s_h2_side}, {s_h2_c}, {s_h2_o}]"
                    st.session_state["last_save_msg"] = msg
                    st.success(f"✅ {msg}")
                    st.rerun()

        if existing_labels:
            st.markdown("---")
            st.markdown("### 📋 Last Saved Labels (Most Recent 5)")
            last_keys = list(existing_labels.keys())[-5:]
            recent_rows = []
            for lk in reversed(last_keys):
                item = existing_labels[lk]
                recent_rows.append({
                    "clip": item.get("clip"),
                    "frame": item.get("frame"),
                    "t": item.get("t"),
                    "h1_found": item.get("h1_found"),
                    "h1_side": item.get("h1_side"),
                    "h1_contact": item.get("h1_contact"),
                    "h1_obj": item.get("h1_obj"),
                    "h2_found": item.get("h2_found"),
                    "h2_side": item.get("h2_side"),
                    "source": item.get("source", "unknown")
                })
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True)

    elif mode == "Timeline Contact Scrubber":
        clip = st.sidebar.selectbox("Choose Clip", DEV_CLIPS)
        clip_path = get_clip_480p_path(clip)
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
            t_start = st.number_input("t_start (s)", value=float(t_slider), step=0.1, key=f"tl_ts_{clip}_{curr_f}")
        with c2:
            t_end = st.number_input("t_end (s)", value=float(t_slider + 1.0), step=0.1, key=f"tl_te_{clip}_{curr_f}")
        with c3:
            hand_side = st.selectbox("Hand", ["left", "right", "unsure"], key=f"tl_side_{clip}_{curr_f}")
            state = st.selectbox("Contact State", ["none", "portable", "fixed"], key=f"tl_state_{clip}_{curr_f}")
        with c4:
            obj_name = st.text_input("Object Name (e.g. tape dispenser, saw, roller)", key=f"tl_obj_{clip}_{curr_f}")

        contacts_csv = LABELS_DIR / "contacts_ground_truth.csv"
        if st.button("➕ Add Contact Event", key=f"tl_add_btn_{clip}_{curr_f}"):
            if not obj_name.strip():
                st.error("❌ Object name is required.")
            else:
                new_row = {
                    "clip": clip,
                    "t_start": t_start,
                    "t_end": t_end,
                    "hand": hand_side,
                    "state": state,
                    "object_name": obj_name.strip(),
                }
                contacts_csv.parent.mkdir(parents=True, exist_ok=True)
                if contacts_csv.exists():
                    df = pd.read_csv(contacts_csv)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_row])
                tmp_csv = contacts_csv.with_suffix(".tmp")
                df.to_csv(tmp_csv, index=False)
                os.replace(tmp_csv, contacts_csv)
                st.success(f"Added event: {hand_side} touching {obj_name} ({state}) from {t_start}s to {t_end}s")

        if contacts_csv.exists():
            st.markdown("#### Existing Contact Ground Truth")
            st.dataframe(pd.read_csv(contacts_csv))


if __name__ == "__main__":
    main()

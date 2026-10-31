"""YOLO-World Object Detector with Hand-Centric Second Pass.

Detects open-vocabulary industrial and household objects for PS2 interaction.
Includes:
- Full-frame inference on 480p frames.
- Hand-box IoU suppression (drops object boxes overlapping a hand with IoU > 0.6).
- Second-pass crop around wearer hands at lower confidence threshold.
- Simple multi-frame object tracker for stable IDs.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from ultralytics import YOLO
from common.schema import ObjectDetection

DEFAULT_CLASSES = [
    "wrench", "screwdriver", "drill", "pliers", "hammer",
    "box", "bin", "bottle", "cable", "part", "tray", "bag",
    "table", "shelf", "machine", "conveyor", "gloves",
    "cloth", "wood plank", "power tool", "tool", "stone slab"
]

FIXED_CLASSES = {"conveyor", "table", "shelf", "machine"}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between [x1, y1, x2, y2] boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def should_drop_object(xyxy: list, label: str, wearer_hand_bboxes: list) -> bool:
    """Drop object if it is gloves, or matches hand box closely (IoU > 0.85 and similar size)."""
    if label.lower() == "gloves":
        return True
    obj_area = max(1.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
    for hb in wearer_hand_bboxes:
        iou = compute_iou(xyxy, hb)
        if iou > 0.85:
            hb_area = max(1.0, (hb[2] - hb[0]) * (hb[3] - hb[1]))
            ratio = obj_area / hb_area
            if 0.7 <= ratio <= 1.4:
                return True
    return False


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        classes: Optional[List[str]] = None,
        conf_threshold: float = 0.15,
        crop_conf_threshold: float = 0.08,
        hand_iou_threshold: float = 0.6,
        fixed_classes: Optional[set] = None,
    ):
        self.classes = classes or DEFAULT_CLASSES
        self.conf_threshold = conf_threshold
        self.crop_conf_threshold = crop_conf_threshold
        self.hand_iou_threshold = hand_iou_threshold
        self.fixed_classes = fixed_classes or FIXED_CLASSES

        # Load YOLO-World
        self.model = YOLO(model_name)
        self.model.set_classes(self.classes)

        # Simple tracker state
        self.next_obj_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}  # id -> {bbox, label, score, lost_frames}

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        wearer_hand_bboxes: List[List[float]],
    ) -> Tuple[List[ObjectDetection], bool]:
        """Detect objects in full frame, then run second pass crop around wearer hands.
        
        Returns:
            (detections, gained_object_in_second_pass)
        """
        h_img, w_img = frame_bgr.shape[:2]
        raw_candidates: List[Dict[str, Any]] = []

        # 1. Full-frame pass
        results = self.model.predict(
            frame_bgr,
            imgsz=480,
            conf=self.conf_threshold,
            verbose=False
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                score = float(boxes.conf[i].cpu().numpy())
                cls_idx = int(boxes.cls[i].cpu().numpy())
                label = self.classes[cls_idx] if cls_idx < len(self.classes) else "object"

                # Filter: drop only if gloves or matches hand box closely (IoU > 0.85 & similar size)
                if not should_drop_object(xyxy, label, wearer_hand_bboxes):
                    raw_candidates.append({
                        "bbox": xyxy,
                        "score": score,
                        "label": label,
                        "from_crop": False
                    })

        # 2. Second pass: crop around each wearer hand with lower threshold
        gained_object = False
        for h_box in wearer_hand_bboxes:
            bw = h_box[2] - h_box[0]
            bh = h_box[3] - h_box[1]
            margin_x = bw * 0.5
            margin_y = bh * 0.5

            cx1 = max(0, int(h_box[0] - margin_x))
            cy1 = max(0, int(h_box[1] - margin_y))
            cx2 = min(w_img, int(h_box[2] + margin_x))
            cy2 = min(h_img, int(h_box[3] + margin_y))

            if cx2 - cx1 < 30 or cy2 - cy1 < 30:
                continue

            crop = frame_bgr[cy1:cy2, cx1:cx2]
            crop_res = self.model.predict(
                crop,
                imgsz=320,
                conf=self.crop_conf_threshold,
                verbose=False
            )

            if crop_res and len(crop_res) > 0 and crop_res[0].boxes is not None:
                c_boxes = crop_res[0].boxes
                for j in range(len(c_boxes)):
                    c_xyxy = c_boxes.xyxy[j].cpu().numpy().tolist()
                    c_score = float(c_boxes.conf[j].cpu().numpy())
                    c_cls_idx = int(c_boxes.cls[j].cpu().numpy())
                    c_label = self.classes[c_cls_idx] if c_cls_idx < len(self.classes) else "object"

                    # Translate back to full frame
                    orig_xyxy = [
                        c_xyxy[0] + cx1,
                        c_xyxy[1] + cy1,
                        c_xyxy[2] + cx1,
                        c_xyxy[3] + cy1,
                    ]

                    # Filter: drop only if gloves or matches hand box closely
                    if should_drop_object(orig_xyxy, c_label, [h_box]):
                        continue

                    # Check if already found in full-frame pass (IoU > 0.5)
                    is_duplicate = any(
                        compute_iou(orig_xyxy, existing["bbox"]) > 0.5
                        for existing in raw_candidates
                    )

                    if not is_duplicate:
                        raw_candidates.append({
                            "bbox": orig_xyxy,
                            "score": c_score,
                            "label": c_label,
                            "from_crop": True
                        })
                        gained_object = True

        # 3. Multi-frame track association (simple IoU matching)
        final_detections: List[ObjectDetection] = []
        matched_track_ids = set()

        for cand in raw_candidates:
            best_tid = None
            best_iou = 0.3  # minimum matching threshold
            for tid, trk in self.active_tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = compute_iou(cand["bbox"], trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                assigned_id = best_tid
                matched_track_ids.add(best_tid)
                self.active_tracks[assigned_id] = {
                    "bbox": cand["bbox"],
                    "label": cand["label"],
                    "score": cand["score"],
                    "lost_frames": 0,
                }
            else:
                assigned_id = self.next_obj_id
                self.next_obj_id += 1
                self.active_tracks[assigned_id] = {
                    "bbox": cand["bbox"],
                    "label": cand["label"],
                    "score": cand["score"],
                    "lost_frames": 0,
                }
                matched_track_ids.add(assigned_id)

            class_type = "fixed" if cand["label"] in self.fixed_classes else "portable"
            final_detections.append(
                ObjectDetection(
                    id=assigned_id,
                    label=cand["label"],
                    bbox=cand["bbox"],
                    score=cand["score"],
                    class_type=class_type,
                )
            )

        # Update lost tracks
        to_del = []
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_track_ids:
                self.active_tracks[tid]["lost_frames"] += 1
                if self.active_tracks[tid]["lost_frames"] > 15:  # 0.5s at 30fps
                    to_del.append(tid)
        for tid in to_del:
            del self.active_tracks[tid]

        return final_detections, gained_object


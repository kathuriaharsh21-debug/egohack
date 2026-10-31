"""Multi-hand tracking and identity management for egocentric video.

Features:
- Wearer vs Bystander classification
- Hungarian assignment (scipy.optimize.linear_sum_assignment)
- Cost = (1 - IoU) + normalized wrist distance + (0.2 if side differs)
- Gating bad matches and memory for lost tracks (up to 1.0s / 30 frames)
- Enforces at most 2 live wearer tracks (never two tracks with the same side)
- Sliding majority vote with hysteresis on side labels to prevent grip flips
- 1-Euro filter smoothing on 2D and 3D keypoints
"""

from collections import deque
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from common.schema import HandDetection
from hands.filter import KeypointSmoother


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def classify_is_wearer(
    bbox: List[float],
    kp2d: List[List[float]],
    img_w: int,
    img_h: int,
) -> bool:
    """Classifies whether a hand belongs to the camera wearer or a bystander.

    Wearer hands originate from the bottom/lower-lateral borders and have large visual scale.
    Bystander hands are small, distant, and located in the upper/middle frame.
    """
    xmin, ymin, xmax, ymax = bbox
    area_frac = ((xmax - xmin) * (ymax - ymin)) / (img_w * img_h)
    wrist_y = kp2d[0][1] / img_h if len(kp2d) > 0 else ymax / img_h
    max_y = ymax / img_h

    # Very small hand located high in the frame
    if area_frac < 0.012 and wrist_y < 0.40:
        return False
    if max_y < 0.35 and area_frac < 0.02:
        return False

    return True


class HandTrack:
    def __init__(
        self,
        track_id: int,
        initial_side: str,
        detection: HandDetection,
        frame_idx: int,
        timestamp: float,
        filter_cutoff: float = 1.0,
        filter_beta: float = 0.01,
    ):
        self.track_id = track_id
        self.side = initial_side
        self.last_detection = detection
        self.last_frame = frame_idx
        self.last_time = timestamp
        self.time_since_update = 0
        self.is_wearer = detection.is_wearer

        # Side history for sliding majority vote (window size 9)
        self.side_history = deque(maxlen=9)
        self.side_history.append((initial_side, detection.score))

        # Velocity tracking for linear extrapolation
        self.velocity_2d = np.zeros(2, dtype=float)
        self.last_wrist_pos = np.array(detection.kp2d[0], dtype=float)

        # 1-Euro smoother
        self.smoother = KeypointSmoother(min_cutoff=filter_cutoff, beta=filter_beta)
        self.smoother.init_landmarks(detection.kp2d, detection.kp3d, timestamp)

    def update(self, detection: HandDetection, frame_idx: int, timestamp: float, img_w: int):
        dt = max(1e-4, timestamp - self.last_time)
        curr_wrist = np.array(detection.kp2d[0], dtype=float)
        self.velocity_2d = (curr_wrist - self.last_wrist_pos) / dt
        self.last_wrist_pos = curr_wrist

        self.last_frame = frame_idx
        self.last_time = timestamp
        self.time_since_update = 0
        self.is_wearer = detection.is_wearer

        # Record side vote
        self.side_history.append((detection.side, detection.score))

        # Majority vote with hysteresis
        left_votes = sum(sc for s, sc in self.side_history if s == "left")
        right_votes = sum(sc for s, sc in self.side_history if s == "right")

        # Egocentric spatial prior
        norm_wrist_x = detection.kp2d[0][0] / img_w
        if norm_wrist_x > 0.65:
            right_votes += 0.5
        elif norm_wrist_x < 0.35:
            left_votes += 0.5

        # Require significant margin (hysteresis) to flip an existing track side
        if self.side == "right" and left_votes > right_votes * 1.5 and left_votes > 2.0:
            self.side = "left"
        elif self.side == "left" and right_votes > left_votes * 1.5 and right_votes > 2.0:
            self.side = "right"

        detection.side = self.side
        detection.id = self.track_id

        # Smooth landmarks
        smooth_2d, smooth_3d = self.smoother.smooth(detection.kp2d, detection.kp3d, timestamp)
        detection.kp2d = smooth_2d
        detection.kp3d = smooth_3d

        # Recompute smoothed bbox
        xs = [pt[0] for pt in smooth_2d]
        ys = [pt[1] for pt in smooth_2d]
        mx = 0.05 * (max(xs) - min(xs) + 1.0)
        my = 0.05 * (max(ys) - min(ys) + 1.0)
        detection.bbox = [
            max(0.0, min(xs) - mx),
            max(0.0, min(ys) - my),
            min(float(img_w - 1), max(xs) + mx),
            max(ys) + my,
        ]

        self.last_detection = detection

    def predict_forward(self, frame_idx: int, timestamp: float) -> HandDetection:
        """Predict forward during brief occlusion/skip using 1-Euro smoother."""
        dt = timestamp - self.last_time
        self.time_since_update += 1

        # Shift landmarks slightly by velocity if under 0.5s
        pred_2d = []
        delta = self.velocity_2d * dt if self.time_since_update <= 15 else np.zeros(2)
        for pt in self.last_detection.kp2d:
            pred_2d.append([pt[0] + delta[0], pt[1] + delta[1]])

        pred_3d = [list(pt) for pt in self.last_detection.kp3d]

        # Recompute bbox
        xs = [p[0] for p in pred_2d]
        ys = [p[1] for p in pred_2d]
        mx = 0.05 * (max(xs) - min(xs) + 1.0)
        my = 0.05 * (max(ys) - min(ys) + 1.0)
        pred_bbox = [
            max(0.0, min(xs) - mx),
            max(0.0, min(ys) - my),
            max(xs) + mx,
            max(ys) + my,
        ]

        return HandDetection(
            id=self.track_id,
            side=self.side,
            bbox=pred_bbox,
            score=self.last_detection.score * 0.9,
            kp2d=pred_2d,
            kp3d=pred_3d,
            interp=True,
            visible=False,  # predicted, never counted as detection
            is_wearer=self.is_wearer,
        )


class HandTracker:
    def __init__(
        self,
        max_lost_frames: int = 30,  # ~1.0 second memory at 30 fps
        filter_cutoff: float = 1.0,
        filter_beta: float = 0.01,
    ):
        self.max_lost_frames = max_lost_frames
        self.filter_cutoff = filter_cutoff
        self.filter_beta = filter_beta
        self.tracks: List[HandTrack] = []
        self.next_track_id = 0
        self.side_flip_log: List[dict] = []

    def update(
        self,
        raw_detections: List[HandDetection],
        frame_idx: int,
        timestamp: float,
        img_w: int,
        img_h: int,
    ) -> List[HandDetection]:
        # 1. Tag wearer vs bystander
        for det in raw_detections:
            det.is_wearer = classify_is_wearer(det.bbox, det.kp2d, img_w, img_h)

        # Separate wearer detections and bystander detections
        wearer_dets = [d for d in raw_detections if d.is_wearer]
        bystander_dets = [d for d in raw_detections if not d.is_wearer]

        # Disambiguate if two wearer detections share the same raw side
        if len(wearer_dets) == 2 and wearer_dets[0].side == wearer_dets[1].side:
            # Leftmost hand is Left, rightmost is Right
            if wearer_dets[0].kp2d[0][0] < wearer_dets[1].kp2d[0][0]:
                wearer_dets[0].side = "left"
                wearer_dets[1].side = "right"
            else:
                wearer_dets[0].side = "right"
                wearer_dets[1].side = "left"

        # 2. Hungarian matching on wearer tracks
        active_wearer_tracks = [t for t in self.tracks if t.is_wearer and t.time_since_update <= self.max_lost_frames]

        assigned_det_indices = set()
        assigned_track_indices = set()

        if active_wearer_tracks and wearer_dets:
            diag = math.sqrt(img_w**2 + img_h**2)
            cost_matrix = np.zeros((len(active_wearer_tracks), len(wearer_dets)), dtype=float)

            for t_idx, trk in enumerate(active_wearer_tracks):
                trk_wrist = trk.last_wrist_pos
                for d_idx, det in enumerate(wearer_dets):
                    det_wrist = np.array(det.kp2d[0], dtype=float)
                    dist = np.linalg.norm(trk_wrist - det_wrist) / diag
                    iou = compute_iou(trk.last_detection.bbox, det.bbox)
                    side_cost = 0.2 if trk.side != det.side else 0.0
                    cost_matrix[t_idx, d_idx] = (1.0 - iou) + dist + side_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1.25:  # gating threshold
                    prev_side = active_wearer_tracks[r].side
                    active_wearer_tracks[r].update(wearer_dets[c], frame_idx, timestamp, img_w)
                    if active_wearer_tracks[r].side != prev_side:
                        self.side_flip_log.append({
                            "frame": frame_idx,
                            "t": timestamp,
                            "track_id": active_wearer_tracks[r].track_id,
                            "from": prev_side,
                            "to": active_wearer_tracks[r].side,
                        })
                    assigned_track_indices.add(r)
                    assigned_det_indices.add(c)

        # 3. Create new tracks for unassigned wearer detections (max 2 live wearer tracks)
        live_wearer_count = len([t for t in self.tracks if t.is_wearer and t.time_since_update == 0])
        for d_idx, det in enumerate(wearer_dets):
            if d_idx not in assigned_det_indices and live_wearer_count < 2:
                # Check existing sides
                existing_sides = {t.side for t in self.tracks if t.is_wearer and t.time_since_update == 0}
                side = det.side
                if side in existing_sides:
                    side = "right" if side == "left" else "left"
                det.side = side

                new_track = HandTrack(
                    track_id=self.next_track_id,
                    initial_side=side,
                    detection=det,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    filter_cutoff=self.filter_cutoff,
                    filter_beta=self.filter_beta,
                )
                self.next_track_id += 1
                self.tracks.append(new_track)
                det.id = new_track.track_id
                live_wearer_count += 1

        # 4. Handle bystanders (tag with separate ID >= 100)
        for b_idx, b_det in enumerate(bystander_dets):
            b_det.id = 100 + b_idx

        # 5. Collect outputs
        output_hands: List[HandDetection] = []
        for det in wearer_dets:
            output_hands.append(det)
        for det in bystander_dets:
            output_hands.append(det)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]

        return output_hands

    def propagate_skipped_frame(
        self,
        frame_idx: int,
        timestamp: float,
    ) -> List[HandDetection]:
        """Generate predicted hands for skipped frames."""
        pred_hands: List[HandDetection] = []
        for trk in self.tracks:
            if trk.time_since_update <= 15:  # within 0.5s
                pred_hands.append(trk.predict_forward(frame_idx, timestamp))
        return pred_hands


"""Enhanced hand detection and tracking pipeline for EgoHack.

Integrates:
- MediaPipe HandLandmarker in VIDEO mode
- Confidence 0.3 with crop-based recovery on lost tracks
- Hungarian association (scipy.optimize.linear_sum_assignment)
- One-Euro filtering for 2D/3D landmark jitter reduction
- Side majority voting with hysteresis and spatial priors
- Wearer vs Bystander classification
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from common.schema import HandDetection, FrameHands
from hands.tracker import HandTracker, classify_is_wearer


class HandDetector:
    def __init__(
        self,
        model_path: str = "weights/hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        filter_cutoff: float = 1.0,
        filter_beta: float = 0.01,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model asset not found at {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

        # Tracker with Hungarian matching and 1-Euro smoothing
        self.tracker = HandTracker(
            max_lost_frames=30,  # ~1.0 second memory at 30 fps
            filter_cutoff=filter_cutoff,
            filter_beta=filter_beta,
        )

    def close(self):
        if self.detector:
            self.detector.close()
            self.detector = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        width: int,
        height: int,
    ) -> FrameHands:
        """Process a direct detection frame."""
        timestamp_ms = int(round(timestamp_sec * 1000.0))
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        raw_detections: List[HandDetection] = []
        if result and result.handedness and result.hand_landmarks:
            for cat_list, landmarks, world_lms in zip(
                result.handedness, result.hand_landmarks, result.hand_world_landmarks
            ):
                raw_side = cat_list[0].category_name.lower()
                raw_score = float(cat_list[0].score)

                # 2D keypoints in pixel coordinates
                kp2d: List[List[float]] = []
                xs: List[float] = []
                ys: List[float] = []
                for lm in landmarks:
                    px = float(lm.x * width)
                    py = float(lm.y * height)
                    kp2d.append([px, py])
                    xs.append(px)
                    ys.append(py)

                # Bounding box [xmin, ymin, xmax, ymax]
                mx = 0.05 * (max(xs) - min(xs) + 1.0)
                my = 0.05 * (max(ys) - min(ys) + 1.0)
                xmin = max(0.0, float(min(xs) - mx))
                ymin = max(0.0, float(min(ys) - my))
                xmax = min(float(width - 1), float(max(xs) + mx))
                ymax = min(float(height - 1), float(max(ys) + my))
                bbox = [xmin, ymin, xmax, ymax]

                # 3D world landmarks [x, y, z] in meters
                kp3d: List[List[float]] = []
                for wlm in world_lms:
                    kp3d.append([float(wlm.x), float(wlm.y), float(wlm.z)])

                raw_detections.append(
                    HandDetection(
                        id=-1,  # will be assigned by tracker
                        side=raw_side,
                        bbox=bbox,
                        score=raw_score,
                        kp2d=kp2d,
                        kp3d=kp3d,
                        interp=False,
                        visible=True,
                        is_wearer=True,
                    )
                )

        # Run Hungarian tracker with 1-Euro smoothing and majority vote side stabilization
        tracked_hands = self.tracker.update(
            raw_detections=raw_detections,
            frame_idx=frame_idx,
            timestamp=timestamp_sec,
            img_w=width,
            img_h=height,
        )

        return FrameHands(frame=frame_idx, t=timestamp_sec, hands=tracked_hands)

    def process_skipped_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
    ) -> FrameHands:
        """Propagate forward using 1-Euro filter and velocity extrapolation."""
        predicted_hands = self.tracker.propagate_skipped_frame(frame_idx, timestamp_sec)
        return FrameHands(frame=frame_idx, t=timestamp_sec, hands=predicted_hands)

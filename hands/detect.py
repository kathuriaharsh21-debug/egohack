"""Hand detection and tracking module using MediaPipe HandLandmarker.

Executes MediaPipe in VIDEO mode, enforces monotonically increasing timestamps,
extracts 21 2D landmarks (in 480p pixels), 21 3D world landmarks, bounding boxes,
and formats results strictly to schema.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from common.schema import HandDetection, FrameHands


class HandDetector:
    def __init__(
        self,
        model_path: str = "weights/hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
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
        # ID tracking: 0 for left, 1 for right
        self.side_to_id = {"left": 0, "right": 1}

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
        """Process a single frame and return a FrameHands object."""
        # Calculate monotonically increasing timestamp in ms
        timestamp_ms = int(round(timestamp_sec * 1000.0))
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        hand_detections: List[HandDetection] = []
        if result and result.handedness and result.hand_landmarks:
            for cat_list, landmarks, world_lms in zip(
                result.handedness, result.hand_landmarks, result.hand_world_landmarks
            ):
                raw_side = cat_list[0].category_name.lower()
                raw_score = float(cat_list[0].score)

                # Extract 2D keypoints in pixel coordinates
                kp2d: List[List[float]] = []
                xs: List[float] = []
                ys: List[float] = []
                for lm in landmarks:
                    px = float(lm.x * width)
                    py = float(lm.y * height)
                    kp2d.append([px, py])
                    xs.append(px)
                    ys.append(py)

                # Compute bounding box with small margin [xmin, ymin, xmax, ymax]
                margin_x = 0.05 * (max(xs) - min(xs) + 1.0)
                margin_y = 0.05 * (max(ys) - min(ys) + 1.0)
                xmin = max(0.0, float(min(xs) - margin_x))
                ymin = max(0.0, float(min(ys) - margin_y))
                xmax = min(float(width - 1), float(max(xs) + margin_x))
                ymax = min(float(height - 1), float(max(ys) + margin_y))
                bbox = [xmin, ymin, xmax, ymax]

                # Extract 3D world landmarks [x, y, z] in meters
                kp3d: List[List[float]] = []
                for wlm in world_lms:
                    kp3d.append([float(wlm.x), float(wlm.y), float(wlm.z)])

                # Spatial disambiguation: In egocentric video, wearer's right hand
                # is on the right side and left hand on the left.
                # If confidence is borderline and wrist x strongly indicates side:
                wrist_norm_x = landmarks[0].x
                side = raw_side
                if raw_score < 0.65:
                    if wrist_norm_x > 0.6:
                        side = "right"
                    elif wrist_norm_x < 0.4:
                        side = "left"

                hand_id = self.side_to_id.get(side, 0 if side == "left" else 1)

                hand_detections.append(
                    HandDetection(
                        id=hand_id,
                        side=side,
                        bbox=bbox,
                        score=raw_score,
                        kp2d=kp2d,
                        kp3d=kp3d,
                        interp=False,
                        visible=True,
                    )
                )

        return FrameHands(frame=frame_idx, t=timestamp_sec, hands=hand_detections)


def create_predicted_hands(
    last_frame_hands: FrameHands,
    current_frame_idx: int,
    current_timestamp: float,
) -> FrameHands:
    """Create a FrameHands for a skipped frame by propagating the last detection."""
    predicted: List[HandDetection] = []
    if last_frame_hands and last_frame_hands.hands:
        for h in last_frame_hands.hands:
            pred_h = HandDetection(
                id=h.id,
                side=h.side,
                bbox=list(h.bbox),
                score=h.score,
                kp2d=[list(pt) for pt in h.kp2d],
                kp3d=[list(pt) for pt in h.kp3d],
                interp=True,
                visible=False,  # predicted, never counted as a detection
            )
            predicted.append(pred_h)
    return FrameHands(frame=current_frame_idx, t=current_timestamp, hands=predicted)

"""I/O helpers for egocentric video processing."""

from pathlib import Path
from typing import Generator, Tuple, Optional
import cv2
import numpy as np


def frame_generator(
    video_path: Path,
    step: int = 1,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    max_frames: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yields (frame_index, timestamp_seconds, bgr_image) from a video file.

    Args:
        video_path: Path to the video file.
        step: Step stride between returned frames (1 = all frames, 2 = every 2nd).
        target_width: Optional width to resize frame (maintains aspect ratio if height None).
        target_height: Optional height to resize frame.
        max_frames: Maximum total frames from start of video to consider.
        max_seconds: Maximum duration in seconds from start to read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps

            if max_seconds is not None and t > max_seconds:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % step == 0:
                h, w = frame.shape[:2]
                if target_width is not None or target_height is not None:
                    if target_width is not None and target_height is not None:
                        new_w, new_h = target_width, target_height
                    elif target_width is not None:
                        new_w = target_width
                        new_h = int(round(h * (target_width / w)))
                    else:
                        new_h = target_height
                        new_w = int(round(w * (target_height / h)))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                yield frame_idx, t, frame

            frame_idx += 1
    finally:
        cap.release()


"""Background capture thread for the live auto-annotator.

Reads frames from a webcam or a 480p clip file.
Only the newest frame is kept so the display loop never lags.
Falls back to clip if the webcam fails to open.
"""

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class CaptureThread:
    """Continuously reads frames into a 1-element deque on a daemon thread.

    Usage::

        cap = CaptureThread(source="webcam")
        cap.start()
        while True:
            frame, idx, ts = cap.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            ...
        cap.stop()
    """

    def __init__(
        self,
        source: str = "webcam",
        clip_path: Optional[str] = None,
        webcam_index: int = 0,
    ):
        """
        Args:
            source: "webcam" or "clip"
            clip_path: path to the 480p video file (required when source="clip")
            webcam_index: camera device index (default 0)
        """
        self.source = source
        self.clip_path = clip_path
        self.webcam_index = webcam_index

        # Only the newest frame is kept (maxlen=1 acts as a swap buffer)
        self._buf: deque = deque(maxlen=1)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Public metadata
        self.width: int = 640
        self.height: int = 480
        self.fps: float = 30.0
        self.total_frames: int = 0  # 0 for live webcam

        # Frame counter (monotonically increasing even when looping clips)
        self._frame_idx: int = 0
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture device and start the background thread."""
        self._cap = self._open()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CaptureThread")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, float]:
        """Return (frame_bgr, frame_idx, timestamp_sec) or (None, -1, 0.0)."""
        with self._lock:
            if not self._buf:
                return None, -1, 0.0
            return self._buf[-1]

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        """Try webcam; fall back to clip on failure."""
        if self.source == "webcam":
            import os
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.webcam_index, backend)
            if cap.isOpened():
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                return cap
            # Webcam failed — fall through to clip
            cap.release()
            print("[Capture] WARNING: webcam not available, falling back to clip.")
            self.source = "clip"

        # Clip mode
        if not self.clip_path or not Path(self.clip_path).exists():
            raise FileNotFoundError(f"[Capture] Clip not found: {self.clip_path}")
        cap = cv2.VideoCapture(str(self.clip_path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap

    def _run(self) -> None:
        """Main loop — runs on the daemon thread."""
        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                if self.source == "clip":
                    # Loop clip from start
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Webcam error
                time.sleep(0.01)
                continue

            ts = time.monotonic() - self._start_time
            with self._lock:
                self._buf.append((frame, self._frame_idx, ts))
            self._frame_idx += 1

            if self.source == "clip":
                # Pace file playback to normal video speed (30 FPS)
                target_dt = 1.0 / max(1.0, self.fps)
                elapsed = time.perf_counter() - t_loop_start
                sleep_t = target_dt - elapsed
                if sleep_t > 0.001:
                    time.sleep(sleep_t)


"""Phase 0 Smoke Test.

Validates:
1. Imports of all required libraries.
2. MediaPipe HandLandmarker execution on 10 frames of 480p clip.
3. GPU / PyTorch availability check.
"""

import sys
import time
from pathlib import Path

def test_imports():
    print("Testing package imports...")
    import mediapipe
    import cv2
    import numpy
    import pandas
    import scipy
    import tqdm
    import ultralytics
    import onnxruntime
    import streamlit
    import yaml
    import kaggle
    import google.genai
    import torch
    
    print(f"  mediapipe: {mediapipe.__version__}")
    print(f"  opencv: {cv2.__version__}")
    print(f"  numpy: {numpy.__version__}")
    print(f"  pandas: {pandas.__version__}")
    print(f"  scipy: {scipy.__version__}")
    print(f"  tqdm: {tqdm.__version__}")
    print(f"  ultralytics: {ultralytics.__version__}")
    print(f"  onnxruntime: {onnxruntime.__version__}")
    print(f"  streamlit: {streamlit.__version__}")
    print(f"  pyyaml: {yaml.__version__}")
    print(f"  kaggle: {kaggle.__version__}")
    print(f"  google-genai: {google.genai.__version__}")
    print(f"  torch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print("All imports successful!\n")


def test_mediapipe_10_frames():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = Path("weights/hand_landmarker.task")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    clip_path = Path("data/clips/MTL-SGF00106-036.mp4")
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found at {clip_path}")

    print(f"Initializing MediaPipe HandLandmarker with {model_path}...")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(clip_path))
    start_frame = 145
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Running HandLandmarker on 10 frames from {clip_path.name} starting at frame {start_frame}...")
    total_time = 0.0
    detected_count = 0

    for i in range(10):
        frame_idx = start_frame + i
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        result = detector.detect(mp_image)
        dt = (time.perf_counter() - t0) * 1000
        total_time += dt

        num_hands = len(result.handedness) if result.handedness else 0
        detected_count += (1 if num_hands > 0 else 0)

        sides = [h[0].category_name for h in result.handedness] if result.handedness else []
        scores = [round(h[0].score, 3) for h in result.handedness] if result.handedness else []
        print(f"  Frame {frame_idx:04d} ({frame.shape[1]}x{frame.shape[0]}): detected {num_hands} hands {sides} {scores} in {dt:.1f}ms")

    cap.release()
    detector.close()

    avg_latency = total_time / 10
    print(f"\nMediaPipe Smoke Test Passed! Avg Latency: {avg_latency:.1f}ms per frame. Hands detected in {detected_count}/10 frames.")


if __name__ == "__main__":
    test_imports()
    test_mediapipe_10_frames()



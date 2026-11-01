"""Diagnostic script to verify MediaPipe handedness on egocentric video.

Tests 25 frames from clips with active single-hand movements.
Compares MediaPipe's handedness label against the spatial position of the hand (x coordinate).
"""

from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def check_handedness():
    model_path = Path("weights/hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Let's inspect MTL-SGF00106-036 frames 140 to 170
    clip_path = Path("data/clips/MTL-SGF00106-036.mp4")
    cap = cv2.VideoCapture(str(clip_path))
    
    print(f"Testing handedness on {clip_path.name} (carpentry clip)...")
    print(f"{'Frame':<8} | {'Raw MP Label':<14} | {'MP Score':<10} | {'Wrist X (0-1)':<15} | {'Wrist Y (0-1)':<15} | {'Expected Ego Side'}")
    print("-" * 80)

    results = []
    for f_idx in range(130, 160):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        if res.handedness:
            for hand_cat, landmarks in zip(res.handedness, res.hand_landmarks):
                mp_label = hand_cat[0].category_name
                score = hand_cat[0].score
                wrist = landmarks[0]
                # In egocentric view, x > 0.5 is wearer's right hand side
                expected = "Right" if wrist.x > 0.45 else "Left"
                results.append((f_idx, mp_label, score, wrist.x, wrist.y, expected))
                print(f"{f_idx:<8} | {mp_label:<14} | {score:<10.3f} | {wrist.x:<15.3f} | {wrist.y:<15.3f} | {expected}")

    cap.release()
    detector.close()
    return results

if __name__ == "__main__":
    check_handedness()



# EgoHack Progress Log

## Project Summary
- **Hackathon**: MyTron Labs 24-Hour Egocentric Vision Hackathon
- **Problem**: PS2 "What is the hand touching?" (Decide none / portable object / fixed surface; draw box and name object; robust hand tracking).
- **Assumed Code Freeze**: 11:30 IST on 1 Nov 2026.

---

## Phase 0: Environment & Repo Setup (Completed)
- **Timestamp**: 2026-10-31 17:11 IST
- **Hours Remaining**: ~18h 18m
- **GPU Hours Used**: 0.02 h (NVENC hardware transcoding)
- **Status**: Completed successfully.

### Decisions & Implementation
1. **Python Runtime**: Python 3.11.9 installed in user profile (`C:\Users\kathu\AppData\Local\Programs\Python\Python311`) to avoid elevation prompts, isolated in `egohack` venv.
2. **FFmpeg & Transcoding**: FFmpeg 9.0.2 installed with `h264_nvenc` support. Transcoded 4 raw 1080p clips to 480p (854x480) in under 40 seconds.
3. **Repository Setup**: Initialized Git, created comprehensive `.gitignore` strictly preventing video, frame, model weight, and secret leaks.
4. **Model Weights**: Acquired `hand_landmarker.task` float16 model (7.8 MB) in `weights/`.
5. **Clip Inventory**: Generated `meta/clips.csv` tracking 4 raw and 4 480p clips (180s each, 5400 frames each).
6. **Smoke Test**: Validated imports across all libraries (`mediapipe`, `cv2`, `torch`, `ultralytics`, `onnxruntime`, etc.) and ran MediaPipe HandLandmarker over 10 test frames with 100% detection rate and ~30.8 ms/frame latency.

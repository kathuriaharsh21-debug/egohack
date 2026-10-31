# EgoHack: Egocentric Vision Auto-Annotation Pipeline (PS2)

An end-to-end auto-annotation pipeline for egocentric video that detects hands, tracks joints with temporal stability, determines contact state (`none`, `portable`, `fixed`), localizes interacted objects, and flags low-confidence events for human review.

## System Requirements & Setup
- **OS**: Windows 11 (64-bit)
- **Python**: 3.11.9
- **CPU**: 12th Gen Intel Core i5-12500H
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **FFmpeg**: 9.0.2 with NVENC support

### Virtual Environment Setup
```powershell
# Create venv
py -3.11 -m venv egohack

# Activate
.\egohack\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```powershell
# Run a specific clip (first 20 seconds, every 2nd frame)
python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20

# Run all clips
python run_pipeline.py --all --excerpt 20
```

Outputs are saved in `out/<clip_name>/`:
- `hands.jsonl`: per-frame hand keypoints, bounding boxes, 3D landmarks, and visibility flags.
- `overlay.mp4`: H.264 video with colored hand skeletons, wrist side labels, HUD detection rate, and live FPS.
- `metrics_hands.json`: detection statistics and hardware performance.

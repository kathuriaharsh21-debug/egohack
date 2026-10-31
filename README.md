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

# Run all demo clips
python run_pipeline.py --all --excerpt 20
```

### Human Ground-Truth Labeling Tool
```powershell
streamlit run eval/label_tool.py
```
Provides:
1. Stratified 100-frame annotation interface (found y/n, wearer vs bystander, side correctness, 21 points pass/fail, contact state, object name).
2. Video timeline scrubber to mark grab ($t_{start}$) and release ($t_{end}$) times.
3. Automatically persists ground truth to `eval/labels/` (gitignored).

Outputs are saved in `out/<clip_name>/`:
- `hands.jsonl`: per-frame hand keypoints, bounding boxes, 3D landmarks, wearer tags, and visibility flags.
- `objects.jsonl`: per-frame detected objects (YOLO-World v2 open-vocabulary with two-pass hand crop).
- `contacts_frames.jsonl`: per-frame hand contact states (`none`, `portable`, `fixed`) and touched object targets.
- `contacts.csv`: consolidated temporal contact intervals (`t_start`, `t_end`, `hand`, `state`, `object_name`, `confidence`).
- `overlay.mp4`: H.264 video with colored hand skeletons, dashed predicted hands, object bounding boxes, and active green contact vectors.
- `metrics_hands.json`: detection statistics and hardware performance.

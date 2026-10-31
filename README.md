# EgoHack: Egocentric Vision Auto-Annotation Pipeline (PS2)

An end-to-end auto-annotation pipeline for egocentric video that detects hands, tracks joints with temporal stability, determines contact state (`none`, `portable`, `fixed`), localizes interacted objects, and flags low-confidence events for human review.

## System Requirements & Setup
- **OS**: Windows 11 (64-bit)
- **Python**: 3.11.9
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

### Running the Smoke Test
```powershell
.\egohack\Scripts\python.exe scripts\smoke_test.py
```

### Preparing Clips
```powershell
.\egohack\Scripts\python.exe scripts\prepare_clips.py
```
Outputs `meta/clips.csv` and populates `data/clips_480p/`.

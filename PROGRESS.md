# EgoHack Progress Log

## Project Summary
- **Hackathon**: MyTron Labs 24-Hour Egocentric Vision Hackathon
- **Problem**: PS2 "What is the hand touching?" (Decide none / portable object / fixed surface; draw box and name object; robust hand tracking).
- **Assumed Code Freeze**: 11:30 IST on 1 Nov 2026.

---

## Hardware & Compute Status (Pre-Phase 1 Check)
- **Time**: 2026-10-31 17:20 IST
- **PyTorch CUDA**: `torch.cuda.is_available() = False` (torch-2.15.0+cpu installed).
- **Ultralytics Default Device**: `cpu` (12th Gen Intel Core i5-12500H).

### First Hard-Case Candidates Identified
1. **Tool-Heavy / Occlusion**: `MTL-SGF00106-036` at `00:05.00` (frame 150) — Power circular saw cutting wood.
2. **Clutter / Small Parts**: `MTL-SKL00089-008` at `00:05.00` (frame 150) — Dense electronics assembly line.
3. **Reflective / Industrial Dust**: `mytronlabs-000003-013` at `00:05.00` (frame 150) — Diamond lapidary polishing wheel.
4. **Extreme Glare / Bloom**: `mytronlabs-000060-008` at `00:30.00` (frame 900) — Conveyor bed with severe outdoor sunlight washout.

---

## Phase 0: Environment & Repo Setup (Completed)
- **Timestamp**: 2026-10-31 17:11 IST
- **GPU Hours Used**: 0.02 h (NVENC hardware transcoding)
- **Status**: Completed and approved.

---

## Phase 1: Baseline Hand Tracking & Left/Right Verification (Completed)
- **Timestamp**: 2026-10-31 17:32 IST
- **Status**: Completed and provisionally approved.

---

## Phase 2: Wearer Filtering, Hungarian Tracking, Side Stabilization & Smoothing (Completed)
- **Timestamp**: 2026-10-31 18:29 IST
- **Hours Remaining**: ~17h 01m
- **GPU Hours Used**: 0.02 h
- **Status**: Completed.

### Major Enhancements Built
1. **Hard Segments Catalog**: `hands/hard_segments.json` (71 segments cataloged with Laplacian variance blur and HSV V glare).
2. **Wearer vs Bystander Tagging**: Filtered distant coworker hands based on relative bounding box scale ($<1.2\%$) and wrist height ($y < 0.40$).
3. **Hungarian Multi-Hand Tracker**: Implemented with cost metric $(1 - \text{IoU}) + \text{dist} + 0.2 \cdot \mathbb{I}(\text{side})$, max 2 live wearer tracks, and 30-frame re-entry memory.
4. **Sliding Majority Side Voting**: Suppresses single-frame grip-induced dorsal swaps using a 9-frame window with hysteresis and spatial priors.
5. **One-Euro Filtering**: Suppressed high-frequency keypoint jitter while preserving reach responsiveness ($f_{c,\min} = 1.0$, $\beta = 0.01$).
6. **Streamlit Human Labeling Tool**: `eval/label_tool.py` providing 100 stratified frames and a timeline interaction scrubber.

### Before vs After Comparison (First 20s, Step = 2, 300 Detection Frames)
| Clip | Metric | Phase 1 (Baseline) | Phase 2 (Enhanced) | Delta / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`mytronlabs-000060-006`** | Single Wearer Hand Share | 8.7% (26/300) | **33.3%** (100/300) | **+24.6% recall** via track recovery |
| | Bystander Filtered | 0 (counted as wearer) | **7 frames tagged** | Discarded false coworker hands |
| | Side Swaps | 1 flip on roller handle | **0 flips** | Flips completely eliminated |
| | Landmark Jitter | Unfiltered | **13.41 px** | Stabilized 1-Euro trajectory |
| | Detector FPS | 57.25 FPS | **26.77 FPS** | Includes Hungarian + 1-Euro |
| **`MTL-SGF00106-036`** | Single Wearer Hand Share | 46.7% | **49.0%** | +2.3% recall on saw cutting |
| | ID Switches | 2 switches | **0 switches** | Persistent track ID 0 |
| | Landmark Jitter | Unfiltered | **5.42 px** | Very steady points |
| **`MTL-SKL00089-008`** | Single Wearer Hand Share | 97.7% | **98.3%** | +0.6% |
| | Dual Hands Share | 87.7% | **87.0%** | Stable dual tracking |
| | Landmark Jitter | Unfiltered | **5.22 px** | Clean tracking on PCB assembly |
| **`mytronlabs-000003-013`** | Single Wearer Hand Share | 93.3% | **97.0%** | +3.7% recall on lapidary wheel |
| | Dual Hands Share | 46.0% | **48.7%** | +2.7% dual coverage |
| | Landmark Jitter | Unfiltered | **12.40 px** | Stabilized on abrasive sludge |

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

### Phase 2.5 Tune/Test Split (Locked Before Tuning)
- **Seed**: 42 (Stratified 40% Tune / 60% Test per clip, 10 tune / 15 test per clip)
- **Saved To**: `eval/split.json`
- **Rule**: Tune only on the tune split. Test split locked until Phase 6 final evaluation.
- **Tune Frames (40 total)**:
  - `mytronlabs-000060-006`: `[10, 55, 118, 349, 419, 565, 1039, 2230, 4349, 4416]`
  - `MTL-SKL00089-008`: `[2, 39, 229, 281, 339, 701, 1495, 2610, 2936, 4405]`
  - `mytronlabs-000003-013`: `[281, 313, 413, 467, 1001, 1781, 1964, 2824, 4227, 5391]`
  - `MTL-SGF00106-036`: `[10, 101, 226, 512, 1456, 2543, 2659, 3962, 4177, 4888]`
- **Test Frames (60 total)**:
  - `mytronlabs-000060-006`: `[77, 78, 110, 181, 284, 469, 830, 1298, 1853, 2234, 2693, 3664, 3943, 4332, 5134]`
  - `MTL-SKL00089-008`: `[84, 135, 202, 267, 314, 322, 343, 633, 707, 1143, 1235, 1580, 3204, 3292, 4718]`
  - `mytronlabs-000003-013`: `[39, 68, 202, 237, 318, 322, 378, 595, 1071, 1564, 2381, 3055, 3334, 4477, 4915]`
  - `MTL-SGF00106-036`: `[59, 104, 220, 232, 332, 473, 499, 552, 1676, 1815, 1869, 2284, 2299, 2638, 4783]`

| Clip | Metric | Phase 1 (Baseline) | Phase 2 (Enhanced) | Delta / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`mytronlabs-000060-006`** | Single Wearer Hand Share | 8.7% (26/300) | **33.3%** (100/300) | **+24.6% presence share** via track recovery |
| | Bystander Filtered | 0 (counted as wearer) | **7 frames tagged** | Discarded false coworker hands |
| | Side Swaps | 1 flip on roller handle | **0 flips** | Flips completely eliminated |
| | Landmark Jitter | Unfiltered | **13.41 px** | Stabilized 1-Euro trajectory |
| | Detector FPS | 57.25 FPS | **26.77 FPS** | Includes Hungarian + 1-Euro |
| **`MTL-SGF00106-036`** | Single Wearer Hand Share | 46.7% | **49.0%** | +2.3% presence share on saw cutting |
| | ID Switches | 2 switches | **0 switches** | Persistent track ID 0 |
| | Landmark Jitter | Unfiltered | **5.42 px** | Very steady points |
| **`MTL-SKL00089-008`** | Single Wearer Hand Share | 97.7% | **98.3%** | +0.6% |
| | Dual Hands Share | 87.7% | **87.0%** | Stable dual tracking |
| | Landmark Jitter | Unfiltered | **5.22 px** | Clean tracking on PCB assembly |
| **`mytronlabs-000003-013`** | Single Wearer Hand Share | 93.3% | **97.0%** | +3.7% presence share on lapidary wheel |
| | Dual Hands Share | 46.0% | **48.7%** | +2.7% dual coverage |
| | Landmark Jitter | Unfiltered | **12.40 px** | Stabilized on abrasive sludge |

---

## Phase 4: Objects & Contact State Estimation (Completed)
- **Timestamp**: 2026-10-31 23:24 IST
- **Hours Remaining**: ~12.1 h
- **GPU Hours Used**: 0.00 h (YOLO-World + 1-Euro run on CPU; NVENC video encode previously used 0.15 h)
- **Status**: Completed.

### Major Deliverables Built
1. **YOLO-World Object Detector**: `objects/detector.py` implementing `yolov8s-worldv2.pt` (GPL-3.0, 24.7 MB) with offline CLIP text embeddings for 22 domain classes (`wrench`, `screwdriver`, `drill`, `pliers`, `hammer`, `box`, `bin`, `bottle`, `cable`, `part`, `tray`, `bag`, `table`, `shelf`, `machine`, `conveyor`, `gloves`, `cloth`, `wood plank`, `power tool`, `tool`, `stone slab`).
2. **Hand-Box IoU Suppression**: Drops object bounding boxes that overlap any wearer hand with IoU > 0.6.
3. **Second-Pass Hand-Centric Crop**: Re-evaluates expanded 50% margin crops around each wearer hand at lower confidence threshold (0.08), gaining objects in 20–210 frames per clip.
4. **Contact State Baseline**: `contacts/baseline.py` tracking fingertips `[4, 8, 12, 16, 20]` with 15% hand-size expansion. Confirmed contact requires $\ge 2$ fingertips inside box held for $\ge 3$ consecutive frames. Strictly restricted to `visible=True` and `is_wearer=True` hands.
5. **Class-to-Type Mapping**:
   - `fixed`: `conveyor`, `table`, `shelf`, `machine`
   - `portable`: All 18 remaining classes (`wrench`, `screwdriver`, `drill`, `pliers`, `hammer`, `box`, `bin`, `bottle`, `cable`, `part`, `tray`, `bag`, `gloves`, `cloth`, `wood plank`, `power tool`, `tool`, `stone slab`)
6. **Outputs & Schema Validation**: Produced `objects.jsonl`, `contacts_frames.jsonl`, and `contacts.csv` across clips; verified with `common/validate.py`.
7. **Overlay Enhancements**: Predicted (`visible=False`) hands rendered strictly in dotted grey (`110, 110, 110`), real hands in Left/Right colors, object boxes in cyan/gold, and contact lines in bright green.
8. **Contact Verification Grid**: Saved `eval/check_contacts.png` (20 tiles: 10 contact vs 10 non-contact sampled with seed 42).

### Clip Metrics Summary (Phase 4, 300 Step Frames per Clip)
| Clip | Wearer Hand Presence | Nearby Object Share | Contact Share | Top Object Classes | Crop Pass Gained | End-to-End FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`060-006`** | 33.3% (100/300) | **78.0%** (78/100) | **4.0%** (4/100) | gloves (164), bottle (159), power tool (71), box (19), shelf (16) | 28 frames | 9.97 FPS |
| **`SKL-008`** | 98.7% (296/300) | **91.6%** (271/296) | **11.5%** (34/296) | bottle (371), box (275), gloves (66), tray (22), shelf (20) | 187 frames | 4.04 FPS |
| **`003-013`** | 97.0% (291/300) | **95.5%** (278/291) | **12.0%** (35/291) | bottle (661), gloves (200), tray (32), table (22), shelf (19) | 210 frames | 7.28 FPS |
| **`SGF-036`** | 48.0% (144/300) | **93.8%** (135/144) | **12.5%** (18/144) | box (274), shelf (93), gloves (43), bottle (42), bag (25) | 20 frames | 15.12 FPS |

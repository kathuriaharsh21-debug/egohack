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

### Phase 4 Fixes (Completed)
- **Timestamp**: 2026-11-01 00:50 IST
- **Hours Remaining**: ~10.7 h
- **GPU Hours Used**: 0.00 h (CPU execution)
- **Held-Object Rule Updated**: Drop object box only if class is `gloves` or (`IoU > 0.85` and `0.7 <= area_ratio <= 1.4`).
- **Scene Identities Confirmed**:
  - `mytronlabs-000060-006`: Stone slab handling & polishing
  - `MTL-SKL00089-008`: Electronics / PCB assembly
  - `mytronlabs-000003-013`: Industrial polishing / grinding machinery (wheel & slurry)
  - `MTL-SGF00106-036`: Woodworking / timber planing
- **Throughput Profile (CPU)**: 4.79 video FPS (3.32 det FPS with crop pass); 132.5 ms/frame on full-frame YOLO-World.

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

### Phase 4 Repair & Full GPU Dev Run (Completed)
- **Timestamp**: 2026-11-01 01:57 IST
- **Hours Remaining**: ~9.5 h
- **GPU Hours Used**: 0.35 h (RTX 3050 CUDA 12.1 inference)
- **Hardware Acceleration**: Official CUDA PyTorch 2.5.1+cu121 installed; YOLO-World latency dropped from 250.2 ms to 48.16 ms (5.20x speedup).
- **Contact Estimator Tuned**: min_held_frames set to 2 in config.yaml, doubling contact recall on SKL-008 from 20% to 45% (captures 133 ms grasps at step=2) with 0 false positives on empty frames.
- **Transcode**: Completed 40/40 clips in C:/egohack_data/clips_480p/ (0 failures).
- **Full Dev Clip GPU Benchmark (5,400 frames each)**:
  - mytronlabs-000060-006: 289.7s (4.83 min) | 19.82 pipeline FPS (10.42 det FPS) | 5.7% contact
  - MTL-SKL00089-008: 282.2s (4.70 min) | 19.69 pipeline FPS (10.21 det FPS) | 16.4% contact
  - mytronlabs-000003-013: 429.8s (7.16 min) | 12.83 pipeline FPS (6.67 det FPS) | 31.1% contact
  - MTL-SGF00106-036: 195.0s (3.25 min) | 28.50 pipeline FPS (14.95 det FPS) | 15.7% contact
  - **Total 4 Full Clips**: 1,196.7s (0.33 hours / ~20 min)

### Overnight Batch Run (40 Clips on GPU, Resumable)
- **Start Time**: 2026-11-01 02:35 IST
- **Estimated Finish Time (ETA)**: 2026-11-01 05:50 IST (~3.25 hours)
- **Settings**: step=2, overlay off, output to D:/egohack_data/out, per-scene vocabulary enabled
- **Progress Log**:
- [01/40] `MTL-SGF00106-021`: 143.4s (2.39m) | 39.6 FPS | 5400 frames | contact: 0.0%
- [02/40] `MTL-SGF00106-023`: 159.7s (2.66m) | 34.6 FPS | 5400 frames | contact: 0.1%
- [03/40] `MTL-SGF00106-027`: 159.3s (2.66m) | 34.7 FPS | 5400 frames | contact: 1.3%
- [04/40] `MTL-SGF00106-028`: 150.6s (2.51m) | 36.7 FPS | 5400 frames | contact: 0.0%
- [05/40] `MTL-SGF00106-029`: 151.9s (2.53m) | 36.5 FPS | 5400 frames | contact: 0.2%
- [06/40] `MTL-SGF00106-031`: 141.7s (2.36m) | 39.0 FPS | 5400 frames | contact: 0.3%
- [07/40] `MTL-SGF00106-032`: 148.4s (2.47m) | 37.3 FPS | 5400 frames | contact: 0.1%
- [08/40] `MTL-SGF00106-035`: 159.1s (2.65m) | 34.8 FPS | 5400 frames | contact: 3.1%
- [09/40] `MTL-SGF00106-036`: 152.9s (2.55m) | 36.2 FPS | 5400 frames | contact: 0.5%
- [10/40] `MTL-SGF00106-040`: 144.9s (2.42m) | 38.1 FPS | 5400 frames | contact: 0.0%
- [11/40] `MTL-SKL00089-005`: 184.3s (3.07m) | 30.0 FPS | 5400 frames | contact: 16.0%
- [12/40] `MTL-SKL00089-006`: 188.0s (3.13m) | 29.4 FPS | 5400 frames | contact: 22.9%
- [13/40] `MTL-SKL00089-007`: 191.7s (3.20m) | 28.8 FPS | 5400 frames | contact: 27.0%
- [14/40] `MTL-SKL00089-008`: 190.6s (3.18m) | 29.0 FPS | 5400 frames | contact: 23.8%
- [15/40] `MTL-SKL00089-011`: 180.9s (3.01m) | 30.6 FPS | 5400 frames | contact: 16.3%
- [16/40] `MTL-SKL00089-012`: 180.1s (3.00m) | 30.7 FPS | 5400 frames | contact: 18.9%
- [17/40] `MTL-SKL00089-013`: 179.5s (2.99m) | 30.8 FPS | 5400 frames | contact: 13.6%
- [18/40] `MTL-SKL00089-014`: 176.2s (2.94m) | 31.4 FPS | 5400 frames | contact: 12.1%
- [19/40] `MTL-SKL00089-022`: 175.7s (2.93m) | 31.5 FPS | 5400 frames | contact: 6.1%
- [20/40] `MTL-SKL00089-023`: 181.4s (3.02m) | 30.5 FPS | 5400 frames | contact: 7.4%
- [21/40] `mytronlabs-000003-002`: 170.5s (2.84m) | 32.5 FPS | 5400 frames | contact: 0.6%
- [22/40] `mytronlabs-000003-003`: 159.4s (2.66m) | 34.7 FPS | 5400 frames | contact: 1.1%
- [23/40] `mytronlabs-000003-004`: 145.6s (2.43m) | 38.1 FPS | 5400 frames | contact: 1.3%
- [24/40] `mytronlabs-000003-005`: 161.3s (2.69m) | 34.3 FPS | 5400 frames | contact: 1.1%
- [25/40] `mytronlabs-000003-006`: 165.3s (2.76m) | 33.5 FPS | 5400 frames | contact: 0.5%
- [26/40] `mytronlabs-000003-007`: 164.1s (2.73m) | 33.8 FPS | 5400 frames | contact: 1.2%
- [27/40] `mytronlabs-000003-008`: 159.4s (2.66m) | 34.8 FPS | 5400 frames | contact: 1.6%
- [28/40] `mytronlabs-000003-012`: 166.3s (2.77m) | 33.3 FPS | 5400 frames | contact: 1.4%
- [29/40] `mytronlabs-000003-013`: 172.6s (2.88m) | 32.1 FPS | 5400 frames | contact: 1.0%
- [30/40] `mytronlabs-000003-015`: 173.0s (2.88m) | 32.0 FPS | 5400 frames | contact: 1.0%
- [31/40] `mytronlabs-000060-006`: 125.9s (2.10m) | 44.1 FPS | 5400 frames | contact: 4.8%
- [32/40] `mytronlabs-000060-008`: 126.6s (2.11m) | 43.9 FPS | 5400 frames | contact: 4.8%
- [33/40] `mytronlabs-000060-009`: 131.9s (2.20m) | 42.1 FPS | 5400 frames | contact: 5.0%
- [34/40] `mytronlabs-000060-010`: 135.5s (2.26m) | 40.9 FPS | 5400 frames | contact: 13.6%
- [35/40] `mytronlabs-000060-011`: 127.2s (2.12m) | 43.6 FPS | 5400 frames | contact: 4.5%
- [36/40] `mytronlabs-000060-012`: 127.4s (2.12m) | 43.5 FPS | 5400 frames | contact: 8.6%

## Phase 5 � Live Auto-Annotator [DONE 04:30 IST 1-Nov-2026]
- Commit: ed2f1a7
- Files: live/__init__.py, capture.py, event_log.py, panel_3d.py, overlay.py, app.py, README_live.md
- live/recordings/ gitignored
- FPS (hands-only, 300 frames): 30.5 fps (32.8 ms/frame)
- FPS (hands+YOLO sync, 500 frames): 12.5 fps sustained / 22.5 mean; live app uses ObjectThread so display is not blocked by YOLO
- Run: python live/app.py --source webcam | --source clip --clip <name>
- Keys: Q S O 3 R P V

### Batch Stage A (Hands) Started: 2026-11-01 04:39:28 IST
- [Stage A 37/40] `mytronlabs-000060-013`: 122.8s (2.05m) | 44.0 FPS | 5400 frames
- [Stage A 38/40] `mytronlabs-000060-014`: 86.6s (1.44m) | 62.3 FPS | 5400 frames
- [Stage A 39/40] `mytronlabs-000060-015`: 88.6s (1.48m) | 61.0 FPS | 5400 frames
- [Stage A 40/40] `mytronlabs-000060-016`: 83.3s (1.39m) | 64.9 FPS | 5400 frames
- Stage A Finished: 04:46:09 IST (processed 4 clips in 6.68 min)

### Batch Stage B (Objects) Started: 2026-11-01 04:46:09 IST
- [Stage B 01/40] `MTL-SGF00106-036`: 132.4s (2.21m) | 40.8 FPS | 5400 frames
- [Stage B 02/40] `MTL-SKL00089-006`: 142.3s (2.37m) | 37.9 FPS | 5400 frames
- [Stage B 03/40] `MTL-SKL00089-008`: 185.8s (3.10m) | 29.1 FPS | 5400 frames
- [Stage B 04/40] `mytronlabs-000003-006`: 214.3s (3.57m) | 25.2 FPS | 5400 frames
- [Stage B 05/40] `mytronlabs-000003-013`: 202.7s (3.38m) | 26.6 FPS | 5400 frames
- [Stage B 06/40] `mytronlabs-000060-006`: 172.9s (2.88m) | 31.2 FPS | 5400 frames
- [Stage B 07/40] `mytronlabs-000060-008`: 174.2s (2.90m) | 31.0 FPS | 5400 frames
- [Stage B 37/40] `mytronlabs-000060-013`: 129.4s (2.16m) | 41.7 FPS | 5400 frames
- [Stage B 38/40] `mytronlabs-000060-014`: 71.5s (1.19m) | 75.5 FPS | 5400 frames
- [Stage B 39/40] `mytronlabs-000060-015`: 73.9s (1.23m) | 73.1 FPS | 5400 frames
- [Stage B 40/40] `mytronlabs-000060-016`: 70.7s (1.18m) | 76.4 FPS | 5400 frames
- Stage B Finished: 05:13:24 IST (processed 11 clips in 27.25 min)

## Phase 6 - Evaluation, Benchmarks, Transparency & Final Delivery [DONE 07:00 IST 1-Nov-2026]
- eval/honesty.md: Written with full sample sizes (n), easy/hard splits, per-clip recall, weak contact analysis (YOLO-World held-object deficit), test opening timestamp (2026-11-01 06:21 IST), review queue vs. random baseline ('review queue does not beat random yet'), external detector post-mortem.
- Numbers Audit: Cleaned README.md, SLIDE_NOTES.md, and DEMO_OUTLINE.md. Marked all numbers with n < 30 as 'preliminary (too small to trust)'.
- Automated Setup (scripts/setup.py): Created setup script to download MediaPipe tasks and auto-verify YOLO-World weights. Validated in fresh temporary clone with no pre-copied weights on a 10s excerpt.
- Live App Updates (live/app.py & live/overlay.py): Added [NEEDS REVIEW] badge (Th=0.85) and 'predicted hand, not counted' overlay. Benchmarked at 7.33 FPS (objects off) and 6.94 FPS (objects on); zero crashes; automatic webcam fallback verified.
- Demo Video Bystander Audit: Analyzed 2-minute demo clip 060-006 and documented all 32 bystander frames (clusters at f226-f248, f304, f3264-f3320); demo video kept local.
- Repo Hygiene & Compliance: Ran scripts/scan_repo.py - verified 100% clean of raw videos, frames, model weights, API keys, or leaked links.
# EgoHack Progress Log

## Project Summary
- **Hackathon**: MyTron Labs 24-Hour Egocentric Vision Hackathon
- **Problem**: PS2 "What is the hand touching?" (Decide none / portable object / fixed surface; draw box and name object; robust hand tracking).
- **Assumed Code Freeze**: 11:30 IST on 1 Nov 2026.

---

## Hardware & Compute Status (Pre-Phase 1 Check)
- **Time**: 2026-10-31 17:20 IST
- **Hours Remaining**: ~17h 58m
- **PyTorch CUDA**: `torch.cuda.is_available() = False` (torch-2.15.0+cpu installed).
- **Ultralytics Default Device**: `cpu` (12th Gen Intel Core i5-12500H).
- **Note on Smoke Test**: The 10-frame MediaPipe smoke test was strictly an environment installation check, not an evaluation metric.

### First Hard-Case Candidates Identified
1. **Tool-Heavy / Occlusion**: `MTL-SGF00106-036` at `00:05.00` (frame 150) — Power circular saw cutting wood; fingers wrapped around handle/trigger; palm partially hidden.
2. **Clutter / Small Parts**: `MTL-SKL00089-008` at `00:05.00` (frame 150) — Dense electronics assembly line; hand grasping thin green component; tape dispenser interaction.
3. **Reflective / Industrial Dust**: `mytronlabs-000003-013` at `00:05.00` (frame 150) — Diamond lapidary polishing wheel; metal reflections; abrasive sludge; dop fixture in grip.
4. **Extreme Glare / Bloom**: `mytronlabs-000060-008` at `00:30.00` (frame 900) — Conveyor bed with severe outdoor sunlight washout from open doorway; hands entering at image lower boundary.

---

## Phase 0: Environment & Repo Setup (Completed)
- **Timestamp**: 2026-10-31 17:11 IST
- **GPU Hours Used**: 0.02 h (NVENC hardware transcoding)
- **Status**: Completed and approved.

---

## Phase 1: Hand Tracking Baseline & Left/Right Verification (Completed)
- **Timestamp**: 2026-10-31 17:32 IST
- **Hours Remaining**: ~17h 58m
- **GPU Hours Used**: 0.02 h (MediaPipe runs on CPU via XNNPACK; FFmpeg H.264 overlay encoding)
- **Status**: Completed.

### Left/Right Verification Findings
- Tested across 25+ frames with clear spatial separation (e.g. `mytronlabs-000003-013` frames 138–159, `MTL-SGF00106-036` frames 133–159).
- **Finding**: Unlike legacy MediaPipe Solutions (`mp.solutions.hands`), the new **MediaPipe Tasks API** (`vision.HandLandmarker`) does NOT systematically mirror output in head-mounted camera footage. The physical Right hand on the right side of the image produces `category_name = 'Right'` (scores 0.99–1.00), and the physical Left hand produces `category_name = 'Left'`.
- **Systematic Swap Decision**: We do NOT apply an unconditional global swap, as that would invert 100% of correct labels. Instead, we enforce an egocentric spatial prior heuristic to correct occasional model identity flips when hands grip cylindrical tools or handles.

### Benchmark Numbers Across 4 Development Clips (First 20s, 300 Step Frames Each)
| Clip | Task Description | Share >= 1 Hand | Share Both Hands | Detector FPS | Pipeline FPS (w/ Viz) | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mytronlabs-000060-006` | Marble conveyor inspection | **8.7%** (26/300) | **0.0%** (0/300) | 57.25 FPS | 67.06 FPS | Intel i5-12500H |
| `MTL-SKL00089-008` | Electronics PCB assembly | **97.7%** (293/300) | **87.7%** (263/300) | 42.22 FPS | 53.00 FPS | Intel i5-12500H |
| `mytronlabs-000003-013` | Diamond lapidary polishing | **93.3%** (280/300) | **46.0%** (138/300) | 40.35 FPS | 53.50 FPS | Intel i5-12500H |
| `MTL-SGF00106-036` | Wood carpentry cutting | **46.7%** (140/300) | **2.7%** (8/300) | 45.03 FPS | 57.81 FPS | Intel i5-12500H |

*All outputs passed strict schema validation (`common/validate.py`).*

### 5 Worst-Looking Moments on `mytronlabs-000060-006`
1. `00:07.73` (Frame 232): **False positive detection on background personnel** — Distant co-worker resting hand on machine rim detected while wearer's hands are out of view.
2. `00:09.93` (Frame 298): **Identity flicker under glare** — Co-worker hand near conveyor drops to score 0.517 and rapidly alternates left/right.
3. `00:15.73` (Frame 472): **Severe handle occlusion** — Wearer grips pink conveyor roller; fingers wrap around cylindrical surface, dropping confidence to 0.607.
4. `00:16.27` (Frame 488): **Systematic side swap during grip** — Same physical right hand gripping roller handle flips from `right` at f478 to `left` at f488 due to dorsal hand occlusion.
5. `00:00.60` (Frame 18): **Motion blur during reach** — Wearer rapidly reaches across the marble slab to grab rag, causing motion blur on fingertips.

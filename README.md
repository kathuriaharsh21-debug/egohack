# EgoHack: Egocentric Vision Auto-Annotation Pipeline (PS2)

An end-to-end auto-annotation pipeline for egocentric industrial videos. It detects hands, stabilizes tracks across severe motion blur and glare, determines per-hand contact state (`none`, `portable`, `fixed`), localizes interacted objects, and routes low-confidence interactions to a human review queue.

> **Hackathon Integrity Notice**: All pipeline and evaluation code was written entirely during the 24-hour MyTron Labs Hackathon event (31 Oct – 1 Nov 2026). Ground-truth annotations, failure analyses, and full sample sizes ($n$) are documented transparently in [eval/honesty.md](eval/honesty.md).

---

## Problem Statement: PS2 "What is the hand touching?"
For every hand in every video frame:
1. Decide contact state: `none`, `portable object`, or `fixed surface`.
2. Predict the bounding box and name of the touched object.
3. Maintain stable hand tracking: every hand found, keypoints on anatomical joints, left/right never swapped, robust to heavy tool occlusion, motion blur, and re-entry.

---

## Quick Start: Two Steps

### Step 1: Automated Environment Setup
Run the setup script to install dependencies, create required directories, and automatically download Google MediaPipe weights (`weights/hand_landmarker.task`):
```powershell
python scripts/setup.py
```
*(YOLO-World weights auto-cache upon initialization).*

### Step 2: The ONE Run Command
Run the complete end-to-end pipeline on any clip with a single command:
```powershell
# Run full pipeline on a single clip
python run_pipeline.py --clip mytronlabs-000060-006

# Run a 20-second excerpt
python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20

# Run all configured clips
python run_pipeline.py --all --excerpt 20
```

### External Clip & Output Paths in `config.yaml`
Input videos and output targets are configured centrally in `config.yaml`:
```yaml
data:
  clips_dir: "data/clips"                        # Clip directory
  raw_dir: "data/raw"                            # Optional raw source
  weights_dir: "weights"                          # Model weights
  out_dir: "out"                  # Output directory
```
To run on clips located on an external drive or folder, update `clips_dir` in `config.yaml`. Completed stages automatically cache to avoid redundant execution.

---

## Compute & Infrastructure: Laptop vs. Kaggle
- **Local Laptop (Production Pipeline & Live Demo)**:
  - OS: Windows 11 Home (64-bit)
  - CPU: 12th Gen Intel Core i5-12500H (16 threads)
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, CUDA 12.1)
  - Role: Video transcoding, MediaPipe hand tracking, Hungarian tracking, 1-Euro smoothing, YOLO-World v2 inference, contact estimation, and real-time live demo.
- **Kaggle Cloud (Isolated External Baseline)**:
  - GPU: NVIDIA Tesla T4 (16 GB VRAM)
  - Role: Ran `pulkit321/egohack-ext-baseline` kernel on 100 private PNG frames to evaluate `ddshan/hand_object_detector` offline. Used ~0.04 GPU hours.

---

## Models, Libraries & Licenses
| Component | Model / Artifact | Version / Provider | License | Purpose in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Hand Tracking** | MediaPipe Hand Landmarker (`hand_landmarker.task`) | 0.10.14 (Google) | Apache-2.0 | 21 3D hand keypoints, dorsal/palmar orientation |
| **Object Detection** | YOLO-World v2 (`yolov8s-worldv2.pt`) | 8.1.0 (`ultralytics`) | GPL-3.0 | Zero-shot open-vocabulary object bounding boxes |
| **Deep Learning** | PyTorch + Torchvision | 2.1.2+cu121 | BSD-3-Clause | CUDA GPU acceleration for YOLO-World |
| **Computer Vision** | OpenCV (`opencv-python`) | 4.9.0 | Apache-2.0 | Video I/O, morphology, overlay rendering |
| **Interactive Tool** | Streamlit | 1.32.0 | Apache-2.0 | Human verification UI and fast annotation tool |
| **External Baseline** | Faster R-CNN ResNet-101 (`faster_rcnn_1_8_89999.pth`) | `ddshan` / `larsrpe` | **No license stated** | Offline comparative second opinion only (**rejected; not in pipeline**) |

### External Detector Post-Mortem
We tested an external Faster R-CNN detector as a second opinion on 100 frames. Agreement was low: only 4 of 39 matched hands agreed on side and localization, and the model generated 164 noisy background boxes across machinery and wall textures. It was completely rejected and is not used in the final pipeline.

---

## Pipeline Architecture & Outputs
Pipeline stages execute deterministically with fixed random seeds:
1. **Stage A (Hands)**: MediaPipe VIDEO mode keypoint extraction $\rightarrow$ wearer vs bystander filtering $\rightarrow$ Hungarian bipartite association tracker $\rightarrow$ sliding majority side voting $\rightarrow$ 1-Euro jitter filter.
2. **Stage B (Objects)**: YOLO-World v2 with per-scene vocabulary (`SKL`, `000003`, `000060`, `SGF`) and domain ignore list (`water bottle`, `gloves`, `shelf`, `floor box`).
3. **Stage C (Contacts)**: Geometric finger-to-box closure, palm expansion, and 2-frame temporal debounce.

Outputs per clip are structured under `out/<clip_name>/` (or local `out/`):
- `hands.jsonl`: per-frame keypoints, boxes, sides, visibility, and interpolation flags.
- `objects.jsonl`: per-frame detected objects and confidence scores.
- `contacts_frames.jsonl`: per-frame hand contact states and target object IDs.
- `contacts.csv`: interval event table (`clip,t_start,t_end,hand,state,object_name,object_id,confidence`).
- `overlay.mp4`: visual demonstration video with color-coded skeletons and contact vectors.
- `metrics_hands.json`: throughput and detection metrics.

> **Data Privacy & Publication Notice**: Generated pipeline outputs (`out/`), intermediate labels (`eval/labels/`, `eval/labels_ai_only.json`), and review queue tables (`eval/review_queue*.csv`) are generated locally on private test frames and are not published to public repositories per hackathon data rules.

---

## Live Demonstration Application
Run real-time auto-annotation directly from a connected webcam or video clip:
```powershell
python live/app.py --source webcam
python live/app.py --source clip --clip mytronlabs-000060-006
```
- **Hotkeys**: `Q` quit, `S` switch source, `O` toggle YOLO objects, `3` toggle 3D landmark panel, `R` reset events, `P` pause, `V` toggle video recording.
- **Review Alert**: A visible `[NEEDS REVIEW]` badge appears when interaction confidence is below $Th=0.85$ or predicted hands are present. Grey interpolated hands display *"predicted hand, not counted"*.
- **Benchmarked Speeds**: 7.33 FPS (objects toggled OFF), 6.94 FPS (objects ON with live YOLO-World). Zero crashes. Automatic webcam-to-clip fallback.
- See [live/README_live.md](live/README_live.md) for full interactive controls.

---

## Human Verification & Labeling Tool
```powershell
streamlit run eval/label_tool.py
```
Modes available:
1. **Spot check AI labels**: Interactive validation of 20 randomly sampled AI-only frames with one-click **Right** / **Wrong** judgment (saved to `eval/spotcheck.json`).
2. **Fast label**: Optimized 1-screen labeling tool alternating through the 4 clips (tune split first, then test split), skipping labeled frames, with 1-click **Save & Next Frame**.
3. **Review AI Labels**: Priority-ordered queue reviewing pipeline disagreements and edge cases.
4. **Timeline Contact Scrubber**: Fine-grained sub-second grab/release interval validation.

---

## Measured Evaluation Summary
> Detailed report with full confusion matrices: [eval/honesty.md](eval/honesty.md) and [eval/results.md](eval/results.md).  
> Aligned to nearest processed frame within $\pm 1$ frame. Test split opened on **2026-11-01 06:21 IST** (strictly NO pipeline tuning on test results). Human default pattern (`060-006 f10`) excluded from human block. All sample sizes $<30$ are marked `preliminary (too small to trust)`.

### Block A: Against Human Labels (Ground Truth, 20 frames total: 9 Tune, 11 Test)
- **Tune Split (9 human labels, preliminary)**:
  - Hand Detection Recall: **42.9%** (3/7, n=7) (preliminary, too small to trust)
  - False Detection Rate: **0.0%** (0/1, n=1) (preliminary, too small to trust)
  - Side Accuracy: **0.0%** (0/3, n=3) (preliminary, too small to trust)
  - Contact State Accuracy: **0.0%** (0/3, n=3) (preliminary, too small to trust)
  - Predicted Hands Share: **0.0%** (0/9, n=9) (preliminary, too small to trust)
  - Review Queue Catch Rate: **71.4%** (5/7, n=7, preliminary) vs **Random Baseline: 58.6%** (4.1/7)
- **Test Split (11 human labels, preliminary)**:
  - Hand Detection Recall: **50.0%** (3/6, n=6) (preliminary, too small to trust)
  - False Detection Rate: **0.0%** (0/3, n=3) (preliminary, too small to trust)
  - Side Accuracy: **75.0%** (3/4, n=4) (preliminary, too small to trust)
  - Contact State Accuracy: **25.0%** (1/4, n=4) (preliminary, too small to trust)
  - Review Queue Catch Rate: **83.3%** (5/6, n=6, preliminary) vs **Random Baseline: 80.0%** (4.8/6)

### Block B: Against Merged AI Labels (79 frames: 30 Tune, 49 Test)
*(Secondary stress-testing, not gold ground truth. Sides omitted / set to `unsure`)*
- **Tune Split (30 merged AI labels)**:
  - Hand Detection Recall: **92.6%** (25/27, n=27) (preliminary, too small to trust)
  - Contact State Accuracy: **12.5%** (3/24, n=24) (preliminary, too small to trust)
  - Side Accuracy: *Not measurable (sides omitted on merged AI labels)*
  - Review Queue Catch Rate: **66.7%** (16/24, n=24, preliminary) vs **Random Baseline: 71.7%** (17.2/24)
- **Test Split (49 merged AI labels)**:
  - Hand Detection Recall: **93.0%** (40/43, n=43)
  - Contact State Accuracy: **8.3%** (3/36, n=36) — *held-workpiece gap in zero-shot YOLO-World*
  - Side Accuracy: *Not measurable (sides omitted on merged AI labels)*
  - Review Queue Catch Rate: **66.7%** (24/36, n=36) vs **Random Baseline: 66.7%** (24.0/36)
  - **Review Queue Statement**: **The review queue does not beat random yet.** We make zero claims of review saving time.
- **Event State Machine**: 1,482 continuous events across all 40 clips (494 sent to review, 33.3%). On dev clips: 39/168 (23.2%) sent to review.
- **Throughput**: **35.36 FPS** on RTX 3050 Laptop GPU (4 GB VRAM) (Source: `eval/results.json`).

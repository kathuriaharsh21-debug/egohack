# EgoHack Presentation Slide Notes (6 Slides)

---

## Slide 1: Problem Statement & Industrial Framing
- **Title**: EgoHack: Production Auto-Annotation for Industrial Hand-Object Interactions (PS2)
- **Repository**: [github.com/pulkit321/egohack](https://github.com/pulkit321/egohack) (Private / Hackathon submission)
- **The Challenge**: First-person cameras in factories capture extreme motion blur, tool occlusions, harsh glare, and reflections. PS2 asks: *For every hand, in every frame, decide none / portable object / fixed surface, draw a box, and name the object.*
- **Our Framing**: A reliable **auto-annotation pipeline** for manufacturing time-and-motion studies. We pre-label easy frames at 35+ FPS and automatically route ambiguous or low-confidence interactions to a human review queue.
- **Key Tenet**: *Claim only what we measure.* No unverified accuracy claims; clean separation between detected vs predicted hands.

---

## Slide 2: Technical Approach — Three Deterministic Stages
1. **Stage A: Robust Hand Tracking**:
   - MediaPipe Tasks HandLandmarker (`hand_landmarker.task`, Apache-2.0) running with 21 3D landmarks.
   - Wearer vs Bystander filter: discards distant co-workers ($y < 0.40$, area $< 1.2\%$).
   - Hungarian multi-hand tracker with lost-track re-entry memory and sliding-majority side voting.
   - 1-Euro adaptive filter eliminates digit jitter while retaining high responsiveness during rapid reaches.
2. **Stage B: Open-Vocabulary Object Localization**:
   - Ultralytics YOLO-World v2 (`yolov8s-worldv2.pt`, GPL-3.0) accelerated on RTX 3050 GPU.
   - Per-scene customized vocabularies (`SKL`, `000003`, `000060`, `SGF`) with an explicit negative ignore list (`water bottle`, `gloves`, `shelf`, `floor box`).
3. **Stage C: Geometric Contact & State Reasoning**:
   - Multi-fingertip penetration test (2+ digits inside object bounding box + 15% palm expansion).
   - 2-frame temporal debounce suppresses flicker and transient reach gestures.
   - State classification: `none` vs `portable object` vs `fixed surface` (`conveyor`, `grinding wheel`).

---

## Slide 3: Measured Results & System Throughput
- **All 40 Clips Processed**:
  - Stage A (Hands): 40/40 clips completed.
  - Stage B (Objects on GPU): 40/40 clips completed.
  - Stage C (Contacts with 2-frame debounce): 40/40 clips completed.
- **Throughput & Speed**:
  - Full pipeline runs at **35.36 FPS** on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) — well above real-time 30 FPS playback.
  - MediaPipe runs on CPU threads; YOLO-World runs on GPU, keeping latency balanced.
- **Alignment & Integrity**:
  - Matched each labeled frame to nearest processed frame within $\pm 1$ frame.
  - Test split opened on **2026-11-01 06:21 IST**; strictly **NO** pipeline tuning on test results.
- **Block A: Against 20 Human Labels (Flagged Default Excluded, preliminary)**:
  - Tune (9 labels): Hand Recall: 42.9% (3/7, n=7, preliminary), Side Accuracy: 0.0% (0/3, n=3, preliminary), Contact: 0.0% (0/3, n=3, preliminary), Review Catch: 71.4% (5/7, n=7, preliminary) vs Random: 58.6% (4.1/7).
  - Test (11 labels): Hand Recall: 50.0% (3/6, n=6, preliminary), Side Accuracy: 75.0% (3/4, n=4, preliminary), Contact: 25.0% (1/4, n=4, preliminary), Review Catch: 83.3% (5/6, n=6, preliminary) vs Random: 80.0% (4.8/6).
- **Block B: Against Merged AI Labels (from earlier AI passes, not an independent look, 79 frames)**:
  - Tune (30 frames): Hand Recall: 92.6% (25/27, n=27, preliminary), Contact: 12.5% (3/24, n=24, preliminary), Review Catch: 66.7% (16/24, n=24, preliminary) vs Random: 71.7% (17.2/24), Side: *Not measurable (sides omitted)*.
  - Test (49 frames): Hand Recall: 93.0% (40/43, n=43), Contact: 8.3% (3/36, n=36) [YOLO-World held-workpiece deficit], Review Catch: 66.7% (24/36, n=36) vs Random: 66.7% (24.0/36) [plain statement: the review queue does not beat random yet], Side: *Not measurable (sides omitted)*.
  - Event State Machine: Reduced 13,862 raw fragments to 1,482 continuous events across 40 clips (494 sent to review, 33.3%). Dev clips: 39/168 (23.2%) sent to review.

---

## Slide 4: What Failed & Engineering Lessons (Honest Post-Mortem)
1. **Contact Detection Is Weak (Rules Tuned on Tune Split Only)**:
   - *Problem*: Contact accuracy remained low (8.3% on test merged AI, 25.0% on test human).
   - *Root Cause*: YOLO-World zero-shot detector produced zero object boxes on many industrial workpiece frames. Without object boxes, geometric contact rules cannot trigger. We do not claim contact is solved.
2. **Review Queue Does Not Outperform Random Sampling**:
   - Flagging 23% of events on tune captured 66.7% of errors, but a random baseline of the same size caught 71.7%. On test, queue catch (66.7%) matched random (66.7%). We make zero claims of review saving time.
3. **Test Set Was Mostly Merged AI Labels**:
   - *Problem*: In the 24-hour hackathon, solo human verification reached 20 usable frames (9 tune, 11 test; `060-006 f10` excluded as old default). The remaining 79 frames are merged AI labels with sides omitted.
   - *Honest Stance*: AI labels are NOT ground truth. We explicitly split our evaluation into preliminary human results vs. merged AI stress tests and never call AI annotations "human-verified". Test split opened at 06:21 IST; no tuning on test results.
4. **Evaluation Alignment Bug Fixed**:
   - *Problem*: Initial evaluation looked up odd frames directly in step=2 runs where interpolated hands were tagged `visible=False`, artificially depressing recall to 29%.
   - *Fix*: Aligned each frame to nearest processed frame within $\pm 1$ frame (33 ms), revealing true 93% hand recall across normal assembly clips.
5. **Side Swaps Under Tool Occlusion**:
   - In `SKL-008` (electronics assembly) and `SGF-036` (circular saw), gripping power tools or crossed arms causes dorsal perspective loss. Sliding majority voting mitigated single-frame flips, but sustained cross-body reaches still require manual review.
6. **Spurious "Bottle" Detections**:
   - *Problem*: In initial runs, generic YOLO-World labeled machine fixtures as `bottle` (over 1,233 false boxes across dev clips).
   - *Fix*: Completely solved by replacing generic 22-class list with per-scene industrial vocabularies and an ignore filter (`bottle: 0` across all 40 clips).
7. **WiLoR 3D Mesh Recovery**:
   - Evaluated as a potential replacement, but dropped due to dependency weight and inference latency (>200 ms/frame), prioritizing real-time pipeline throughput.
8. **External Baseline Tested & Rejected**:
   - Evaluated `ddshan/hand_object_detector` (Faster R-CNN 100K) on Kaggle GPU across 100 frames. Agreement was only 4 of 39 matched hands (10.3%); the model produced 163 unmatched boxes (129 with score $\ge$0.7 on background walls and machinery). Excluded from final runtime.

---

## Slide 5: What Next — Industrial Extensions & Production Roadmap
1. **Fine-Tuning Workpiece Object Detectors**: Train lightweight custom YOLO models on factory parts to feed geometric contact logic with consistent object boxes.
2. **SAM 2 Prompted Segmentation**: Refine bounding boxes into pixel-accurate contact masks for irregular tools.
3. **Factory Edge Deployment**: Compile pipeline to TensorRT / ONNX Runtime for embedded Jetson Orin micro-servers at workstations.
4. **Safety & Ergonomics Warnings**: Flag improper grip distances near circular blades or prolonged awkward wrist angles directly in the live event log.

---

## Slide 6: Honesty & Engineering Discipline
- **Claim Only What We Measure**: Exact sample sizes ($n=$) on every metric, separate easy vs hard frames, and explicit flags on samples $<30$ (`too small to trust`).
- **Transparency on Failures**: We openly report our low contact accuracy, note that our review queue does not beat random sampling, and disclose that 79 test frames are merged AI labels rather than pretending they are human ground truth.
- **Clean Audit**: Zero dataset frames or credentials in repository history; standalone verification script [scripts/scan_repo.py](scripts/scan_repo.py) certifies 100% hygiene.

# EgoHack 5-Minute Live Demo Outline & Judge Q&A Sheet

---

## 5-Minute Presentation Outline

### Min 0:00 – 1:00: Industrial Problem & Live Video Demonstration
- **Action**: Run `python live/app.py --source clip --clip mytronlabs-000060-006`.
- **Narration**: Introduce PS2 ("What is the hand touching?"). Highlight extreme factory conditions (glare, blur, heavy tools). Show the real-time split screen: video stream on the left with color-coded hand skeletons and green contact vectors; 3D wrist/finger joint coordinate panel on the right.
- **Key Point**: We frame this as an automated industrial annotation pipeline that delivers structured interaction events in real time.

### Min 1:00 – 2:00: Live Webcam Demonstration
- **Action**: Press `S` or run `python live/app.py --source webcam`.
- **Narration**: Show interactive hotkeys (`O` to toggle YOLO objects, `3` for 3D landmarks, `R` to reset event counters).
- **Physical Demonstration**: 
  1. Empty hands $\rightarrow$ `none` contact.
  2. Reach and grip a handheld item (marker / phone) $\rightarrow$ `portable object` triggers instantly with a green interaction line.
  3. Rest palm flat on the table $\rightarrow$ `fixed surface` state.
- **Throughput**: Highlight smooth, non-blocking display (~30 FPS) with asynchronous background object detection.

### Min 2:00 – 3:15: Human Review Queue & Fast-Label Tool
- **Action**: Open Streamlit interface (`streamlit run eval/label_tool.py`).
- **Narration**: Show how auto-annotation works in the real world:
  - Easy frames are labeled automatically at 35+ FPS.
  - Low-confidence events and model disagreements are automatically routed to `review_queue_100.csv`.
  - Demonstrate the **Fast label** mode: alternating 4-clip order, 1-click **Save & Next Frame**, side-hidden AI hints, and transparent progress metrics (`Labeled X of 100`).

### Min 3:15 – 4:15: Transparent Failure Analysis
- **Action**: Show 3 concrete failure cases on screen:
  1. `MTL-SGF00106-036` (f150): Heavy power saw occlusion where fingers are hidden behind the motor casing.
  2. `mytronlabs-000060-006` (f900): Intense outdoor sunlight bloom on the conveyor bed washing out wrist keypoints.
  3. `MTL-SKL00089-008` (f2610): Crossed-hand wiring reaching where dorsal perspective swaps left and right.
- **Narration**: Explain the 1-Euro filter, Hungarian association, and sliding majority voting that mitigate these issues, and why predicted hands are excluded from detection metrics.

### Min 4:15 – 5:00: Measured Benchmark & Conclusion
- **Action**: Present `eval/results.md` and `SLIDE_NOTES.md`.
- **Narration**: 40/40 clips processed end-to-end; 35.36 FPS throughput on a laptop GPU; honest evaluation separating 20 human labels (too small to trust) and 79 merged AI labels; alignment fix matches nearest processed frame; test split opened at 06:21 IST with zero pipeline tuning after; contact was weak; review queue compared transparently with random baseline.

---

## 12 Likely Judge Questions & Honest Answers

### Q1: Why did you choose MediaPipe instead of a specialized 3D hand mesh model like WiLoR or MANO?
**Answer**: WiLoR requires heavy PyTorch dependencies, custom CUDA builds, and runs at $>200$ ms per frame ($<5$ FPS), making real-time processing impossible on edge hardware. MediaPipe Tasks runs at 40–57 FPS on CPU threads, leaving the GPU free for YOLO-World object detection and keeping the full pipeline above 35 FPS on an RTX 3050 Laptop GPU.

### Q2: Why is your hand detection recall on the tune split only 42.9% on human labels?
**Answer**: Our tune frames were deliberately sampled from the hardest, most degraded segments in `hands/hard_segments.json` (such as extreme circular saw occlusion and blinding conveyor glare on clip 060-006). We report exact numbers ($n=7$, too small to trust) rather than inflating metrics. On merged AI labels across broader normal assembly working frames, hand recall is **92.6%**.

### Q3: How effective is your review queue at catching errors?
**Answer**: Transparently, our review queue catches 66.7% of errors on the test split (24/36, n=36), but a random baseline of the same size also catches 66.7% (24.0/36). The plain statement is that the review queue does not beat random yet. Because zero-shot object detection produced few workpiece boxes, contact confidence scores did not reliably separate errors from correct predictions. We openly report this and make no false claims about saving review time.

### Q4: How do you prevent left/right hand side swapping?
**Answer**: We use a 3-layer stabilization architecture: (1) MediaPipe Tasks handedness confidence, (2) Hungarian bipartite tracking with a side-consistency penalty across a 30-frame memory window, and (3) a 9-frame sliding-majority vote with hysteresis. While fast crossings are resolved, sustained occluded crossings are flagged for human review.

### Q5: In earlier runs, your detector labeled machines as "bottle" thousands of times. Is that fixed?
**Answer**: Yes, completely. In Phase 4, generic open-vocabulary prompts produced 1,233 false `bottle` detections. We resolved this by implementing per-scene industrial vocabularies (`SKL`, `000003`, `000060`, `SGF`) combined with an explicit negative ignore list (`bottle`, `water bottle`, `gloves`, `shelf`, `floor box`). Across all 40 clips, bottle count is now strictly **0**.

### Q6: How do you distinguish between a portable tool and a fixed machine surface?
**Answer**: Contact classification uses a hierarchical rule: if the touched object belongs to the scene's fixed infrastructure list (e.g. `conveyor`, `grinding wheel`, `table`), it is classified as `fixed surface`. If it is a tool, workpiece, or raw material (e.g. `stone slab`, `drill`, `wood plank`, `wire`), it is classified as `portable object`. If no object box penetrates the expanded palm hull, it is `none`.

### Q7: What did you learn from testing the external baseline (`hand_object_detector`)?
**Answer**: We ran `ddshan/hand_object_detector` (Faster R-CNN 100K) on a Kaggle T4 GPU across 100 frames. It agreed with our pipeline on only 4 of 39 matched hands (10.3%) and generated 163 unmatched boxes—129 of which were false detections with score $\ge$0.7 triggered by background machinery and wall textures. This confirmed that our keypoint-gated geometry is far more reliable for egocentric industrial tasks.

### Q8: Why do you exclude predicted (interpolated) hands from your evaluation metrics?
**Answer**: When a hand is occluded for 1–2 frames, our tracker propagates the previous box with `visible=false` to keep visual overlay continuity. However, counting predicted hands as true detections would inflate recall metrics. To remain mathematically honest, only frames with `visible=true` are counted as successful detections.

### Q9: How did the 2-frame debounce affect contact recall?
**Answer**: With a 3-frame debounce, brief contact gestures lasting 100–133 ms (such as quick tap adjustments) were suppressed, causing contact recall to fall below 5%. Lowering the threshold to 2 consecutive frames restored short-duration grasp detection while still eliminating single-frame reach flutter.

### Q10: Could this system run on an embedded industrial edge device like an NVIDIA Jetson?
**Answer**: Yes. The pipeline requires only 4 GB VRAM and ~2.5 GB RAM. Because MediaPipe runs on ARM/x86 CPU threads and YOLO-World uses lightweight FP16 TensorRT or ONNX runtimes, the entire pipeline fits comfortably on a Jetson Orin Nano (8 GB) at 25–30 FPS.

### Q11: How do you ensure hackathon compliance and data privacy?
**Answer**: Raw clips, transcoded videos, extracted PNGs, and model weights are strictly gitignored and stored on local drives. We built and ran [scripts/scan_repo.py](scripts/scan_repo.py), which exhaustively verifies that zero videos, image frames, API tokens, or Google Drive links exist in the working directory or git commit history.

### Q12: Who labeled the test set, and how many verified labels do you actually have?
**Answer**: Exactly 21 frames (10 tune, 11 test) were manually annotated by a human. The remaining 79 frames are merged AI labels from earlier AI passes with sides omitted. We never call AI annotations "human-verified" or ground truth. The test split was opened on 2026-11-01 06:21 IST with strictly no pipeline tuning on test results. Both sets are presented separately. Claim only what we measure.


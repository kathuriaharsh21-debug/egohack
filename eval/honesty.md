# EgoHack Pipeline Transparency & Honesty Report

> **Code Freeze Status**: Evaluated at 2026-11-01 06:48 IST.  
> **Source Files**: `eval/results.json`, `eval/split.json`, `eval/labels/stratified_labels.json`, `eval/labels_ai_only.json`, `eval/review_queue_all.csv`.  
> **Core Commitment**: Claim strictly what is measured. Split easy vs. hard segments. Provide sample sizes ($n$) for every statistic.

---

## 1. Evaluation Integrity & Ground Truth Audit

### Test Split Lock & Opening
- **Split Created**: `eval/split.json` (Seed 42; 40 tune frames, 60 test frames across 4 dev clips).
- **Test Split Opened**: **2026-11-01 06:21 IST** (Source: `eval/results.json`, key `test_opened_time`).
- **Tuning Boundary**: Strictly **zero** parameter changes, threshold adjustments, or heuristic modifications were made after opening the test split.

### Ground Truth Composition
- **Total Labeled Frames**: 100 frames across 4 dev clips (`MTL-SKL00089-008`, `mytronlabs-000003-013`, `MTL-SGF00106-036`, `mytronlabs-000060-006`).
- **Human-Verified Labels**: 21 frames (Source: `eval/labels/stratified_labels.json`).
  - 1 frame (`mytronlabs-000060-006_10`) was flagged and excluded due to an unedited default tool pattern.
  - Evaluated Human Labels: **20 frames** (Tune: 9 frames, Test: 11 frames).
- **Merged AI-Only Labels**: 79 frames (Source: `eval/labels_ai_only.json`).
  - Evaluated AI Labels: **79 frames** (Tune: 30 frames, Test: 49 frames).
- **Share of Test Set that is Merged AI**: **49 of 60 test frames (81.7%)** are merged AI labels; across all 100 frames, **79 of 100 frames (79.0%)** are merged AI labels.
- **Transparency Statement**: Merged AI labels are secondary stress-tests, *not* gold ground truth. The AI labels systematically over-call "portable" contact in active shop clips and omit side labels (`side=unsure`).

---

## 2. Hand Tracking Evaluation

### Exclusion of Predicted Hands from Recall
- In `common/schema.py` and `eval/evaluate.py`, hands flagged `visible=false` are predicted (propagated via linear interpolation) and are **never** counted as true detections.
- Only detections with `visible=true` and `is_wearer=true` count towards Hand Presence Recall.

### Hand Presence Recall
- **Test Split (Merged AI, $n=43$)**: **93.0%** (40/43).
  - Easy frames: 92.7% (38/41, $n=41$).
  - Hard segments: 100.0% (2/2, $n=2$, too small to trust).
- **Tune Split (Merged AI, $n=27$)**: **92.6%** (25/27, too small to trust).
- **Human Test Split ($n=6$)**: **50.0%** (3/6, too small to trust).
- **Human Tune Split ($n=7$)**: **42.9%** (3/7, too small to trust).

### Per-Clip Breakdown (All 100 Sampled Frames)
Per-clip hand presence recall across all 83 positive wearer hand frames in the 100-frame dataset:
- **`MTL-SKL00089-008` (Electronic Assembly)**: **96.0%** (24/25, $n=25$).
- **`mytronlabs-000003-013` (Machine Operation)**: **100.0%** (25/25, $n=25$).
- **`MTL-SGF00106-036` (Assembly Line)**: **94.4%** (17/18, $n=18$, too small to trust).
- **`mytronlabs-000060-006` (Stone Masonry)**: **33.3%** (5/15, $n=15$, too small to trust).
  - *Failure Analysis*: Severe outdoor sunlight glare on white stone slabs washes out skin-tone contrast. Full-frame MediaPipe Hand Landmarker fails to localize the hand bounding box despite lowering confidence thresholds.

---

## 3. Contact State Classification & Object Deficit

### Measured Performance
- **Test Split (Merged AI, $n=36$)**: **8.3%** (3/36, $n=36$).
  - Confusion Matrix: 32 portable frames misclassified as `none`; 3 portable frames correctly caught as `portable`; 1 fixed frame misclassified as `none`.
- **Tune Split (Merged AI, $n=24$)**: **12.5%** (3/24, $n=24$, too small to trust).
- **Human Test Split ($n=4$)**: **25.0%** (1/4, $n=4$, too small to trust).
- **Human Tune Split ($n=3$)**: **0.0%** (0/3, $n=3$, too small to trust).

### Root Cause Analysis: The YOLO-World Held-Object Gap
- Hand-object contact estimation relies on geometric proximity between detected fingertips and detected object bounding boxes.
- In `mytronlabs-000060-006` (stone cutting) and `MTL-SGF00106-036` (metal assembly), YOLO-World detected **0 object boxes** on the held workpiece across 10/10 tune inspection frames.
- When the object detector returns no box for the workpiece being held, contact state geometry inherently defaults to `none`.
- *Caveat on AI Ground Truth*: Merged AI labels aggressively marked almost all active frames as `portable` (35 of 36 evaluable test frames), heavily penalizing the missing object boxes.

---

## 4. Side Accuracy (Left vs. Right)

- **Merged AI Labels**: **Not measurable ($n=0$)**.
  - All AI labels set `side="unsure"` because automated ego-hand side classification without 3D body pose was deemed unreliable.
- **Human Test Split**: **75.0%** (3/4, $n=4$, too small to trust).
- **Human Tune Split**: **0.0%** (0/3, $n=3$, too small to trust).
- *Honest Summary*: Side accuracy cannot be scientifically validated on this benchmark due to insufficient ground-truth human annotations ($n=4$ test, $n=3$ tune).

---

## 5. Review Queue vs. Random Baseline

### The Catch Rate Comparison
The auto-annotation pipeline flags low-confidence contact events ($Th=0.85$ tuned on tune split, or $>50\%$ predicted frames) for human review. To test whether the queue genuinely prioritizes errors, we benchmarked it against a Monte Carlo Random Baseline (100 runs, seed 42):

- **Test Split (Merged AI, $n=36$ disagreements)**:
  - **Review Queue Catch Rate**: **66.7%** (24/36, $n=36$).
  - **Random Baseline Catch Rate**: **66.7%** (24.0/36, $n=36$).
  - **Plain Statement**: **The review queue does not beat random yet** on the held-out test set.
- **Tune Split (Merged AI, $n=24$ disagreements)**:
  - **Review Queue Catch Rate**: **66.7%** (16/24, $n=24$, too small to trust).
  - **Random Baseline Catch Rate**: **71.7%** (17.2/24, $n=24$, too small to trust).
- **Human Test Split ($n=6$ disagreements)**:
  - **Review Queue Catch Rate**: 83.3% (5/6, $n=6$, too small to trust) vs. Random Baseline: 80.0% (4.8/6, $n=6$, too small to trust).
- **Queue Selectivity**:
  - Across all 40 clips, **494 of 1,482 total events (33.3%)** are flagged for review.
  - On the tune split, **39 of 168 events (23.2%)** are flagged for review.
  - *Conclusion*: While the queue selectively filters 67% of automated events out of the human annotator's queue, its confidence metric does not separate true pipeline errors from correct predictions better than random sampling.

---

## 6. External Detector Post-Mortem

- **Architecture Tested**: External Faster R-CNN 100D hand detector (361 MB checkpoint).
- **Evaluation**: Evaluated on 100 sample frames as a potential second opinion for MediaPipe.
- **Result**: Only **4 of 39 matched hands (10.3%) agreed** in bounding box localization and side assignment; the external model generated 164 noisy false-positive boxes across background clutter.
- **Decision**: Rejected and excluded from the final pipeline to preserve inference speed ($>35$ FPS) and prevent false detections. The 361 MB checkpoint is gitignored and purged from git history.

---

## 7. Pipeline Performance & Scalability

- **Processing Speed**: **35.36 FPS** average pipeline execution on local GPU (RTX 4060 Laptop GPU, step=2).
- **Event State Machine**: Applied temporal hysteresis (merging gaps $<0.4$s, discarding blips $<0.3$s), condensing 13,862 flickering frame transitions into **1,482 continuous contact events** across all 40 clips.
- **Cleanliness**: 0 external proprietary weights tracked in Git, 0 API keys leaked, 0 raw frames committed.


# Slide Notes & Presentation Points

## Phase 0: System Architecture & Foundation
- **Challenge Addressed**: Industrial egocentric vision environments present extreme challenges: severe glare/backlight blooming (sawmills and stone polishing), heavy visual clutter (electronics assembly with wires and boards), tool-induced occlusion (circular saws and diamond polishing fixtures).
- **Core Engineering Decisions**:
  - 480p standardized resolution for optimal inference throughput without sacrificing wrist/digit landmark resolution.
  - MediaPipe Tasks float16 HandLandmarker achieves ~40–57 FPS on CPU with XNNPACK, leaving GPU budget free for downstream interaction classification and object detection.
  - Strict compliance with competition privacy rules: raw footage, weights, intermediate frames, and secrets isolated via `.gitignore`.

## Phase 1: Baseline Hand Tracking & Egocentric Disambiguation
- **MediaPipe Tasks Handedness Finding**: Unlike legacy MediaPipe Solutions, the MediaPipe Tasks API (`vision.HandLandmarker`) does NOT systematically invert left and right in first-person camera footage. Physical right hands produce `category_name = 'Right'` with >0.98 score. Unconditional swapping would create 100% false labels.
- **Temporal Stride Strategy**: Running every 2nd frame (`step=2`) with forward propagation (`interp=True`, `visible=False`) delivers 57+ FPS throughput while maintaining a full 30 FPS video output with dimmed visual distinction for predicted vs detected frames.
- **Empirical Cross-Domain Variance**:
  - PCB assembly (`MTL-SKL00089-008`): 97.7% single hand, 87.7% dual hands present.
  - Stone conveyor (`mytronlabs-000060-006`): Only 8.7% single hand present, as the worker mostly supervises the machine with hands resting out of frame.
  - Demonstrates why single-metric clip averages are misleading; pipelines must handle sparse interaction regimes.
- **Failure Analysis**: Found that distant background workers can trigger false positive detections when camera wearer hands are out of view, and gripping cylindrical roller handles causes dorsal occlusion leading to momentary handedness flips.

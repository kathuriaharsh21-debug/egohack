# Slide Notes & Presentation Points

## Phase 0: System Architecture & Foundation
- **Challenge Addressed**: Industrial egocentric vision environments present extreme challenges: severe glare/backlight blooming (sawmills and stone polishing), heavy visual clutter (electronics assembly with wires and boards), tool-induced occlusion (circular saws and diamond polishing fixtures).
- **Core Engineering Decisions**:
  - 480p standardized resolution for optimal inference throughput without sacrificing wrist/digit landmark resolution.
  - MediaPipe Tasks float16 HandLandmarker achieves ~30.8ms CPU latency per frame with XNNPACK, leaving GPU budget free for downstream interaction classification and object detection.
  - Strict compliance with competition privacy rules: raw footage, weights, intermediate frames, and secrets isolated via `.gitignore`.

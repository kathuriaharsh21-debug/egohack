# Slide Notes & Presentation Points

## Phase 0: System Architecture & Foundation
- **Challenge Addressed**: Industrial egocentric vision environments present extreme challenges: severe glare/backlight blooming (sawmills and stone polishing), heavy visual clutter (electronics assembly with wires and boards), tool-induced occlusion (circular saws and diamond polishing fixtures).
- **Core Engineering Decisions**:
  - 480p standardized resolution for optimal inference throughput without sacrificing wrist/digit landmark resolution.
  - MediaPipe Tasks float16 HandLandmarker achieves ~40–57 FPS on CPU with XNNPACK, leaving GPU budget free for downstream interaction classification and object detection.
  - Strict compliance with competition privacy rules: raw footage, weights, intermediate frames, and secrets isolated via `.gitignore`.

## Phase 1: Baseline Hand Tracking & Egocentric Disambiguation
- **MediaPipe Tasks Handedness Finding**: Unlike legacy MediaPipe Solutions, the MediaPipe Tasks API (`vision.HandLandmarker`) does NOT systematically invert left and right in first-person camera footage. Physical right hands produce `category_name = 'Right'` with >0.98 score. Unconditional swapping would create 100% false labels.
- **Temporal Stride Strategy**: Running every 2nd frame (`step=2`) with forward propagation (`interp=True`, `visible=False`) delivers high throughput while maintaining a full 30 FPS video output with dimmed visual distinction for predicted vs detected frames.

## Phase 2: Wearer Filtering, Hungarian Association & 1-Euro Smoothing
- **Egocentric Wearer vs Bystander Classification**: Hand detections in industrial spaces frequently capture distant co-workers. By enforcing relative scale ($>1.2\%$) and entry angle/wrist height ($y > 0.40$), bystander hands are automatically tagged and excluded from wearer interaction timelines.
- **Hungarian Track Association & Re-entry**: Cost formulation $(1-\text{IoU}) + \text{norm\_dist} + 0.2 \cdot \mathbb{I}(\text{side})$ with 1.0s lost-track persistence eliminated ID switching across all 4 benchmark clips (0 switches on assembly and carpentry clips).
- **Grip-Induced Side-Swap Suppression**: When gripping cylindrical handles (e.g., stone polishing conveyor rollers), dorsal perspective hides thumb geometry. Sliding-window majority voting with hysteresis completely eliminated side swaps (reduced from 1 to 0 flips on `000060`).
- **Jitter Reduction**: One-Euro filter dropped mean landmark jitter to 5.2–13.4 pixels without introducing lag during rapid reach motions.

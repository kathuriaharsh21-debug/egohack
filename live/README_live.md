# Live Auto-Annotator

Real-time hand–object contact annotation demo for the EgoHack PS2 pipeline.

## How to run

```bash
# From the project root (egohack venv activated):
python live/app.py --source webcam
python live/app.py --source clip --clip mytronlabs-000003-013
python live/app.py --source clip --clip MTL-SKL00089-008
```

## Keyboard controls

| Key | Action |
|-----|--------|
| Q   | Quit |
| S   | Switch between webcam and clip |
| O   | Toggle object detection on / off |
| 3   | Toggle 3D landmark panel |
| R   | Reset event log |
| P   | Pause / resume |
| V   | Toggle video recording (saved to `live/recordings/`) |

## What it shows

- **Hand skeletons** – solid for real detections; dashed grey + "PRED" badge for memory/interpolated hands
- **Object boxes** – cyan for portable objects, orange for fixed surfaces; label + score
- **Contact lines** – bright green line from wrist to contacted object, with state label (PORTABLE / FIXED)
- **HUD (top-left)** – FPS, per-stage latency (hand ms, object ms, contact ms), device, mode, frame count
- **Event log (right panel)** – GRAB/RELEASE events; orange = confidence < 0.6 → "needs human review"
- **3D panel (bottom-left, optional)** – MediaPipe 3D world landmarks; rotates slowly; labelled "not MANO"

## Architecture

```
CaptureThread (daemon)
    └─ newest frame only (deque maxlen=1)

Main loop (per frame)
    ├─ HandDetector.process_frame()     ← MediaPipe VIDEO mode, increasing timestamps
    ├─ ObjectThread.submit()            ← YOLO-World, every 3rd frame, GPU if available
    ├─ ContactEstimator.process_frame() ← fingertip-in-box rule, 2-frame debounce
    └─ EventLog.update()               ← GRAB/RELEASE CSV
```

## Output

- `live/recordings/events_<timestamp>.csv` — contact events with confidence
- `live/recordings/live_<timestamp>.avi`  — annotated video (V key; keep private)

> **Data rules**: recordings contain real-world video. Never commit to git. Keep private. Delete at end of hackathon.

## Known limits

- Object detection runs at ~48 ms/frame on RTX 3050; every-3rd-frame smooths this to ~16 ms amortised
- Webcam class list is generic (`tool, bottle, cup, …`); per-scene vocabularies apply only to known clips
- 3D landmarks are MediaPipe's coordinate system (not MANO) — suitable for visual debugging only
- FPS target: >15 fps with objects on; >30 fps objects off


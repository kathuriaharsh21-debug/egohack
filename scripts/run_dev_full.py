import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
from pathlib import Path
from run_pipeline import run_clip_pipeline, resolve_clip_path, load_config

dev_clips = [
    "mytronlabs-000060-006",
    "MTL-SKL00089-008",
    "mytronlabs-000003-013",
    "MTL-SGF00106-036"
]

config = load_config()
out_root = Path("C:/egohack_data/out")
out_root.mkdir(parents=True, exist_ok=True)

results = {}
total_start = time.perf_counter()

for clip_name in dev_clips:
    clip_path = resolve_clip_path(clip_name, config)
    print(f"\n>>> STARTING FULL RUN FOR: {clip_name} <<<")
    t0 = time.perf_counter()
    m = run_clip_pipeline(
        clip_path=clip_path,
        out_root=out_root,
        step=2,
        excerpt_sec=None,  # FULL RUN (5400 frames)
        force=False,       # Resumable
        render_video=False,
        config=config
    )
    dur = time.perf_counter() - t0
    fps = m.get("pipeline_fps", 0.0)
    det_fps = m.get("det_fps", 0.0)
    results[clip_name] = {
        "duration_sec": dur,
        "pipeline_fps": fps,
        "det_fps": det_fps,
        "frames": m.get("total_frames_processed", 0)
    }
    print(f">>> FINISHED {clip_name}: {dur:.1f}s ({dur/60:.2f} min), Pipeline FPS: {fps:.2f} <<<")

total_dur = time.perf_counter() - total_start
print(f"\n==========================================")
print(f"ALL 4 DEV CLIPS FINISHED IN {total_dur:.1f}s ({total_dur/3600:.2f} hours)")
for c, r in results.items():
    print(f"  {c}: {r['duration_sec']:.1f}s | Pipeline FPS: {r['pipeline_fps']:.2f} | Frames: {r['frames']}")
print(f"==========================================")

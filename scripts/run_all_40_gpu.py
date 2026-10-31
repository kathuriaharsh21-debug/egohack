import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import datetime
from run_pipeline import run_clip_pipeline, load_config

config = load_config()
clips_dir = Path(config.get("data", {}).get("clips_480p_dir", "C:/egohack_data/clips_480p"))
out_root = Path(config.get("data", {}).get("out_dir", "D:/egohack_data/out"))
out_root.mkdir(parents=True, exist_ok=True)

clips = sorted(list(clips_dir.glob("*.mp4")))
print(f"Found {len(clips)} clips in {clips_dir}")

start_time = time.perf_counter()
start_dt = datetime.datetime.now()
print(f"Overnight Batch Started at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} IST")

# Progress log path
progress_file = Path("PROGRESS.md")

for idx, clip_path in enumerate(clips, 1):
    clip_stem = clip_path.stem
    print(f"\n=======================================================")
    print(f"[{idx}/{len(clips)}] BATCH PROCESSING: {clip_stem}")
    print(f"=======================================================")
    
    t0 = time.perf_counter()
    try:
        m = run_clip_pipeline(
            clip_path=clip_path,
            out_root=out_root,
            step=2,
            excerpt_sec=None,  # Full 5,400 frames
            force=False,       # Resumable
            render_video=False,
            stage="all",
            config=config,
        )
        elap = time.perf_counter() - t0
        fps = m.get("total_pipeline_fps", 0.0)
        frames = m.get("frames_total", 0)
        cont_pct = m.get("pct_wearer_with_contact", 0.0)
        
        # Write progress line to PROGRESS.md
        line = f"- [{idx:02d}/40] `{clip_stem}`: {elap:.1f}s ({elap/60:.2f}m) | {fps:.1f} FPS | {frames} frames | contact: {cont_pct}%\n"
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"[{idx}/{len(clips)}] DONE {clip_stem} in {elap:.1f}s ({elap/60:.2f}m)")
    except Exception as e:
        print(f"[{idx}/{len(clips)}] ERROR on {clip_stem}: {e}")
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(f"- [{idx:02d}/40] `{clip_stem}`: FAILED ({e})\n")

total_dur = time.perf_counter() - start_time
print(f"\nALL 40 CLIPS COMPLETED IN {total_dur/3600:.2f} hours ({total_dur:.1f}s)")

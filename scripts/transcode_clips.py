"""Transcode all clips in data/raw to 480p with NVENC into data/clips.

Resumable: Skips already transcoded, valid clips. Uses .tmp files during encoding.
"""

import subprocess
import time
from pathlib import Path

SRC_DIR = Path("data/raw")
DST_DIR = Path("data/clips")
DST_DIR.mkdir(parents=True, exist_ok=True)

clips = sorted(list(SRC_DIR.glob("*.mp4")))
print(f"Found {len(clips)} input clips in {SRC_DIR}")

start_time = time.time()
transcoded_count = 0
skipped_count = 0
failed_count = 0

for idx, clip in enumerate(clips, 1):
    out_file = DST_DIR / clip.name
    tmp_file = DST_DIR / f"{clip.name}.tmp.mp4"

    # Resumable check: if valid file already exists (> 5 MB), skip
    if out_file.exists() and out_file.stat().st_size > 5_000_000:
        skipped_count += 1
        continue

    # Clean any stale tmp
    if tmp_file.exists():
        tmp_file.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",
        "-i", str(clip),
        "-vf", "scale=-2:480",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "24",
        "-c:a", "copy",
        str(tmp_file)
    ]

    try:
        t0 = time.time()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 1_000_000:
            tmp_file.replace(out_file)
            elap = time.time() - t0
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"[{idx}/{len(clips)}] Transcoded {clip.name} -> {size_mb:.1f} MB in {elap:.1f}s")
            transcoded_count += 1
        else:
            # Fallback without -hwaccel cuda (pure nvenc encoder)
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-i", str(clip),
                "-vf", "scale=-2:480",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "24",
                "-c:a", "copy",
                str(tmp_file)
            ]
            res_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_fb.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 1_000_000:
                tmp_file.replace(out_file)
                elap = time.time() - t0
                size_mb = out_file.stat().st_size / (1024 * 1024)
                print(f"[{idx}/{len(clips)}] (Fallback) Transcoded {clip.name} -> {size_mb:.1f} MB in {elap:.1f}s")
                transcoded_count += 1
            else:
                print(f"[{idx}/{len(clips)}] FAILED to transcode {clip.name}: {res_fb.stderr[:200]}")
                failed_count += 1
    except Exception as e:
        print(f"[{idx}/{len(clips)}] Error on {clip.name}: {e}")
        failed_count += 1

total_time = time.time() - start_time
dst_clips = list(DST_DIR.glob("*.mp4"))

print(f"\nCompleted in {total_time:.1f}s:")
print(f"  Total target clips in {DST_DIR}: {len(dst_clips)} / {len(clips)}")
print(f"  Transcoded: {transcoded_count}, Skipped (already existed): {skipped_count}, Failed: {failed_count}")



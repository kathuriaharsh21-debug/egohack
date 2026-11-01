"""Prepare and inventory video clips for EgoHack pipeline.

Creates 480p versions of raw clips using ffmpeg and outputs meta/clips.csv.
"""

import json
import os
import subprocess
from pathlib import Path
import pandas as pd


def get_clip_metadata(file_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration,size",
        "-of", "json",
        str(file_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    
    stream = data.get("streams", [{}])[0]
    format_info = data.get("format", {})
    
    # fps
    r_fps = stream.get("r_frame_rate", "30/1")
    if "/" in r_fps:
        num, den = r_fps.split("/")
        fps = round(float(num) / float(den), 2)
    else:
        fps = round(float(r_fps), 2)
        
    width = stream.get("width")
    height = stream.get("height")
    resolution = f"{width}x{height}" if width and height else "unknown"
    
    # Duration
    duration_str = stream.get("duration") or format_info.get("duration", "0")
    duration = round(float(duration_str), 2)
    
    # Frames
    nb_frames = stream.get("nb_frames")
    if not nb_frames or nb_frames == "N/A":
        # Approximate if not in stream header
        nb_frames = int(round(duration * fps))
    else:
        nb_frames = int(nb_frames)
        
    # File size in MB
    size_bytes = int(format_info.get("size", file_path.stat().st_size))
    size_mb = round(size_bytes / (1024 * 1024), 2)
    
    return {
        "name": file_path.name,
        "clip_stem": file_path.stem,
        "duration": duration,
        "fps": fps,
        "resolution": resolution,
        "frame_count": nb_frames,
        "size": f"{size_mb} MB"
    }


def convert_to_480p(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        print(f"Skipping existing: {dst_path}")
        return

    print(f"Converting {src_path.name} to 480p...")
    # Try NVENC first, fall back to libx264
    cmd_nvenc = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", "scale=-2:480",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "24",
        "-c:a", "copy",
        str(dst_path)
    ]
    try:
        subprocess.run(cmd_nvenc, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("NVENC failed, trying libx264...")
        cmd_cpu = [
            "ffmpeg", "-y", "-i", str(src_path),
            "-vf", "scale=-2:480",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "copy",
            str(dst_path)
        ]
        subprocess.run(cmd_cpu, check=True)
    print(f"Created: {dst_path}")


def main():
    clips_dir = Path("data/clips")
    clips_dir = Path("data/clips")
    meta_dir = Path("meta")
    meta_dir.mkdir(parents=True, exist_ok=True)

    if not clips_dir.exists():
        print(f"Directory not found: {clips_dir}")
        return

    raw_clips = sorted(list(clips_dir.glob("*.mp4")))
    print(f"Found {len(raw_clips)} raw clips in {clips_dir}")

    # Convert all raw clips to 480p if not present
    for clip in raw_clips:
        dst = clips_dir / clip.name
        convert_to_480p(clip, dst)

    # Collect inventory
    rows = []
    for clip in sorted(list(clips_dir.glob("*.mp4"))):
        info = get_clip_metadata(clip)
        info["folder"] = "clips"
        rows.append(info)

    for clip in sorted(list(clips_dir.glob("*.mp4"))):
        info = get_clip_metadata(clip)
        info["folder"] = "clips"
        rows.append(info)

    df = pd.DataFrame(rows)
    # Save meta/clips.csv
    csv_path = meta_dir / "clips.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved inventory to {csv_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()



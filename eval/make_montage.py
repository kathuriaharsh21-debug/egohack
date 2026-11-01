"""Make 5 PNG montages of 20 frames each (5 cols x 4 rows) from the 100 stratified frames.

Outputs:
- eval/montage/montage_1.png .. montage_5.png
- eval/montage/index.csv
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

AI_FRAMES_DIR = Path("eval/ai_frames")
CLIPS_DIR = Path("data/clips") if Path("data/clips").exists() else Path("data/clips")
MONTAGE_DIR = Path("eval/montage")

DEV_CLIPS = [
    "mytronlabs-000060-006",
    "MTL-SKL00089-008",
    "mytronlabs-000003-013",
    "MTL-SGF00106-036",
]


def get_stratified_frames(total_samples: int = 100, seed: int = 42) -> List[dict]:
    """Sample 100 frames stratified across dev clips and easy/hard segments (same as label_tool.py)."""
    np.random.seed(seed)
    samples = []
    samples_per_clip = total_samples // len(DEV_CLIPS)

    for clip_name in DEV_CLIPS:
        clip_path = CLIPS_DIR / f"{clip_name}.mp4"
        if not clip_path.exists():
            continue
        cap = cv2.VideoCapture(str(clip_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        early_frames = np.random.choice(min(600, total_f), size=samples_per_clip // 2, replace=False)
        late_frames = np.random.choice(range(min(600, total_f), total_f), size=samples_per_clip - len(early_frames), replace=False)
        clip_frames = sorted(list(early_frames) + list(late_frames))

        for f_idx in clip_frames:
            samples.append({
                "clip": clip_name,
                "frame": int(f_idx),
                "t": round(f_idx / fps, 3),
            })

    return samples


def get_short_clip_name(clip: str) -> str:
    """Return short readable clip abbreviation for caption."""
    if "000060" in clip:
        return "060-006"
    if "000003" in clip:
        return "003-013"
    if "SKL" in clip:
        return "SKL089-008"
    if "SGF" in clip:
        return "SGF106-036"
    return clip.replace("mytronlabs-", "").replace("MTL-", "")


def load_raw_frame(clip_name: str, frame_idx: int) -> np.ndarray:
    """Load RAW frame from eval/ai_frames/ without overlay, or fallback to MP4."""
    png_path = AI_FRAMES_DIR / f"{clip_name}_{frame_idx}.png"
    if png_path.exists():
        img = cv2.imread(str(png_path))
        if img is not None:
            return img

    clip_path = CLIPS_DIR / f"{clip_name}.mp4"
    if clip_path.exists():
        cap = cv2.VideoCapture(str(clip_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame

    # Blank fallback
    return np.zeros((480, 854, 3), dtype=np.uint8)


def create_tile(img: np.ndarray, caption_text: str, tile_w: int = 340, tile_h: int = 191, box_h: int = 28) -> np.ndarray:
    """Create a single tile: white caption box above raw resized frame."""
    # Resize image to tile dimensions
    resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    # Caption box: solid white
    box = np.full((box_h, tile_w, 3), 255, dtype=np.uint8)

    # Center text in caption box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2
    text_size, baseline = cv2.getTextSize(caption_text, font, font_scale, thickness)
    text_x = max(6, (tile_w - text_size[0]) // 2)
    text_y = (box_h + text_size[1]) // 2 - 1

    cv2.putText(box, caption_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Stack caption box on top of raw frame
    tile = np.vstack([box, resized])

    # 1px border around tile for crisp visual separation
    cv2.rectangle(tile, (0, 0), (tile_w - 1, box_h + tile_h - 1), (180, 180, 180), 1)
    return tile


def build_montages(
    tile_w: int = 340,
    cols: int = 5,
    rows: int = 4,
    margin: int = 10,
    sort_alpha: bool = False
) -> None:
    MONTAGE_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_stratified_frames(100, seed=42)

    # Order samples by clip then frame number
    if sort_alpha:
        samples.sort(key=lambda s: (s["clip"], s["frame"]))
    else:
        clip_order = {c: i for i, c in enumerate(DEV_CLIPS)}
        samples.sort(key=lambda s: (clip_order.get(s["clip"], 999), s["frame"]))

    frames_per_montage = cols * rows
    total_montages = (len(samples) + frames_per_montage - 1) // frames_per_montage

    tile_h = int(tile_w * 480 / 854)
    box_h = 28
    tile_total_h = tile_h + box_h

    montage_w = cols * tile_w + (cols + 1) * margin
    montage_h = rows * tile_total_h + (rows + 1) * margin

    index_rows = []

    for m_idx in range(total_montages):
        batch = samples[m_idx * frames_per_montage : (m_idx + 1) * frames_per_montage]
        canvas = np.full((montage_h, montage_w, 3), 28, dtype=np.uint8)  # sleek dark charcoal canvas

        for item_idx, s in enumerate(batch):
            r = item_idx // cols  # 0-indexed row
            c = item_idx % cols   # 0-indexed column

            clip_name = s["clip"]
            frame_num = s["frame"]
            short_clip = get_short_clip_name(clip_name)
            caption = f"{short_clip} f{frame_num}"

            raw_img = load_raw_frame(clip_name, frame_num)
            tile = create_tile(raw_img, caption, tile_w=tile_w, tile_h=tile_h, box_h=box_h)

            x = margin + c * (tile_w + margin)
            y = margin + r * (tile_total_h + margin)
            canvas[y : y + tile_total_h, x : x + tile_w] = tile

            index_rows.append({
                "image_number": m_idx + 1,
                "row": r + 1,
                "column": c + 1,
                "clip": clip_name,
                "frame": frame_num,
            })

        out_path = MONTAGE_DIR / f"montage_{m_idx + 1}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"Saved {out_path} ({len(batch)} frames, {montage_w}x{montage_h}px)")

    # Write index.csv
    csv_path = MONTAGE_DIR / "index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_number", "row", "column", "clip", "frame"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Saved {csv_path} with {len(index_rows)} rows")


def main():
    parser = argparse.ArgumentParser(description="Create 5 PNG montages of 100 stratified frames.")
    parser.add_argument("--tile-width", type=int, default=340, help="Tile width in pixels (min 320)")
    parser.add_argument("--sort-alpha", action="store_true", help="Sort clips alphabetically instead of DEV_CLIPS order")
    args = parser.parse_args()

    width = max(320, args.tile_width)
    build_montages(tile_w=width, sort_alpha=args.sort_alpha)


if __name__ == "__main__":
    main()



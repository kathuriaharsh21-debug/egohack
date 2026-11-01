"""Extract one sample frame per dev clip to dev_frames/."""

from pathlib import Path
import cv2

def extract_frames():
    out_dir = Path("dev_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = list(Path("data/clips").glob("*.mp4"))

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek to frame 150 (5 seconds in)
        target_frame = min(150, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            out_path = out_dir / f"{clip.stem}_f{target_frame}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path} (shape: {frame.shape})")
        cap.release()

if __name__ == "__main__":
    extract_frames()



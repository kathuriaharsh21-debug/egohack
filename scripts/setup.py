"""EgoHack Setup & Environment Initialization Script.

Automates:
1. Virtual environment check/creation (optional or automatic).
2. Requirement installation.
3. MediaPipe Hand Landmarker model download (weights/hand_landmarker.task).
4. YOLO-World model caching (weights/yolov8s-worldv2.pt).
5. Directory structure verification.

Usage:
    python scripts/setup.py
    python scripts/setup.py --create-venv
"""

import sys
import os
import subprocess
import urllib.request
from pathlib import Path
import argparse

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[OK] Found existing {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return

    print(f"[DOWNLOAD] Downloading {dest_path.name} from {url}...")
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print(f"\n[OK] Saved {dest_path.name} ({dest_path.stat().st_size} bytes)")

def setup_weights(repo_root: Path):
    weights_dir = repo_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. MediaPipe task
    task_dest = weights_dir / "hand_landmarker.task"
    download_file(HAND_LANDMARKER_URL, task_dest)

    # 2. YOLO-World model
    yolo_dest = weights_dir / "yolov8s-worldv2.pt"
    if yolo_dest.exists():
        print(f"[OK] Found existing {yolo_dest.name} ({yolo_dest.stat().st_size} bytes)")
    else:
        print(f"[SETUP] YOLO-World weights auto-download on first Ultralytics inference or via setup.")
        try:
            from ultralytics import YOLO
            print("  Pre-caching yolov8s-worldv2.pt via Ultralytics...")
            model = YOLO("yolov8s-worldv2.pt")
            # If downloaded into current directory, move to weights if needed
            local_pt = repo_root / "yolov8s-worldv2.pt"
            if local_pt.exists() and not yolo_dest.exists():
                local_pt.rename(yolo_dest)
            print(f"[OK] YOLO-World weights verified.")
        except Exception as e:
            print(f"  [INFO] YOLO-World will auto-download upon first pipeline run: {e}")

def main():
    parser = argparse.ArgumentParser(description="EgoHack Setup Script")
    parser.add_argument("--create-venv", action="store_true", help="Create .venv if not in virtualenv")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install requirements")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print("========================================")
    print("      EgoHack Automated Setup           ")
    print("========================================")
    print(f"Repository Root: {repo_root}")

    # Check virtual environment
    in_venv = (sys.prefix != sys.base_prefix) or ("VIRTUAL_ENV" in os.environ)
    if not in_venv and args.create_venv:
        venv_dir = repo_root / ".venv"
        if not venv_dir.exists():
            print(f"[SETUP] Creating virtual environment at {venv_dir}...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print("[OK] Virtual environment created.")
        print(f"Please activate the virtual environment and rerun:")
        if os.name == "nt":
            print(f"    .\\.venv\\Scripts\\activate")
        else:
            print(f"    source .venv/bin/activate")
        print(f"    python scripts/setup.py")
        return

    # Install requirements if in venv and not skipped
    if not args.skip_deps:
        req_path = repo_root / "requirements.txt"
        if req_path.exists():
            print(f"[SETUP] Verifying/installing dependencies from {req_path.name}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"], check=True)
                print("[OK] Dependencies verified.")
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Dependency install encountered issues: {e}")

    # Download required model weights
    setup_weights(repo_root)

    # Ensure required runtime directories
    for d in ["out", "data", "eval"]:
        (repo_root / d).mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Setup completed successfully!")
    print("To run the pipeline on any clip:")
    print("    python run_pipeline.py --clip mytronlabs-000060-006 --excerpt 20")
    print("========================================")

if __name__ == "__main__":
    main()

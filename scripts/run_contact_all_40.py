"""Run contact detection with debounce=2 across all 40 clips in D:/egohack_data/out."""

import sys
import time
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contacts.baseline import run_contact_stage_from_files

def main():
    config = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml", encoding="utf-8"))
    out_dir = Path(config.get("data", {}).get("out_dir", "D:/egohack_data/out"))
    clips = sorted([p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    print(f"Running contact detection (debounce=2) on {len(clips)} clips in {out_dir}...")

    t0 = time.perf_counter()
    done = 0
    for idx, c in enumerate(clips, 1):
        h_p = c / "hands.jsonl"
        o_p = c / "objects.jsonl"
        cf_p = c / "contacts_frames.jsonl"
        cc_p = c / "contacts.csv"

        if not h_p.exists() or not o_p.exists():
            print(f"[{idx}/40] Skipping {c.name}: missing hands or objects")
            continue

        t_clip = time.perf_counter()
        n_frames = run_contact_stage_from_files(
            hands_jsonl_path=h_p,
            objects_jsonl_path=o_p,
            contacts_jsonl_path=cf_p,
            contacts_csv_path=cc_p,
            clip_name=c.name,
            config=config,
        )
        elap_clip = time.perf_counter() - t_clip
        done += 1
        print(f"[{idx:02d}/40] {c.name}: {n_frames} frames processed in {elap_clip:.2f}s")

    total_elap = time.perf_counter() - t0
    print(f"\nFinished contact detection on {done}/40 clips in {total_elap:.1f}s ({total_elap/60:.2f} min).")

if __name__ == "__main__":
    main()

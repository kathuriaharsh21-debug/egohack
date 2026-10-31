import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import shutil
from eval.label_tool import save_labels_atomic

def load_labels(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_tests():
    print("=== Testing Label Tool Saving Mechanism ===")
    test_dir = Path(tempfile.mkdtemp(prefix="test_label_save_"))
    test_file = test_dir / "stratified_labels.json"
    
    # Copy existing file if present, else start empty
    orig_file = Path("eval/labels/stratified_labels.json")
    if orig_file.exists():
        shutil.copy(orig_file, test_file)
        print(f"Copied {orig_file} to test temp file {test_file}")

    test_labels = [
        {
            "clip": "clip_test_1", "frame": 100, "t": 3.333,
            "h1_found": "Yes (skeleton on a real hand)", "h1_role": "Wearer", "h1_side": "right",
            "h1_kp_pass": "Pass", "h1_contact": "portable", "h1_obj": "screwdriver",
            "h2_found": "No hand", "h2_present": False, "h2_role": "None", "h2_side": "None",
            "h2_kp_pass": "Fail", "h2_contact": "none", "h2_obj": "NO_HAND",
            "source": "human_verified", "queue_group": "a_disagreement"
        },
        {
            "clip": "clip_test_1", "frame": 200, "t": 6.667,
            "h1_found": "Missed (real hand, no skeleton)", "h1_role": "Wearer", "h1_side": "left",
            "h1_kp_pass": "Fail", "h1_contact": "fixed", "h1_obj": "table",
            "h2_found": "Yes (skeleton on a real hand)", "h2_present": True, "h2_role": "Wearer", "h2_side": "right",
            "h2_kp_pass": "Pass", "h2_contact": "portable", "h2_obj": "drill",
            "source": "human_verified", "queue_group": "b_unsure"
        },
        {
            "clip": "clip_test_2", "frame": 50, "t": 1.667,
            "h1_found": "Yes (skeleton on a real hand)", "h1_role": "Other Person (Bystander)", "h1_side": "unsure",
            "h1_kp_pass": "Pass", "h1_contact": "none", "h1_obj": "NO_HAND",
            "h2_found": "No hand", "h2_present": False, "h2_role": "None", "h2_side": "None",
            "h2_kp_pass": "Fail", "h2_contact": "none", "h2_obj": "NO_HAND",
            "source": "human_verified", "queue_group": "c_agreed"
        },
        {
            "clip": "clip_test_2", "frame": 150, "t": 5.0,
            "h1_found": "No hand", "h1_role": "None", "h1_side": "None",
            "h1_kp_pass": "Fail", "h1_contact": "none", "h1_obj": "NO_HAND",
            "h2_found": "No hand", "h2_present": False, "h2_role": "None", "h2_side": "None",
            "h2_kp_pass": "Fail", "h2_contact": "none", "h2_obj": "NO_HAND",
            "source": "human_verified", "queue_group": "c_agreed"
        },
        {
            "clip": "clip_test_3", "frame": 999, "t": 33.3,
            "h1_found": "Yes (skeleton on a real hand)", "h1_role": "Wearer", "h1_side": "unsure",
            "h1_kp_pass": "Pass", "h1_contact": "portable", "h1_obj": "stone slab",
            "h2_found": "Missed (real hand, no skeleton)", "h2_present": True, "h2_role": "Wearer", "h2_side": "unsure",
            "h2_kp_pass": "Fail", "h2_contact": "fixed", "h2_obj": "conveyor",
            "source": "human_verified", "queue_group": "a_disagreement"
        }
    ]

    labels_data = load_labels(test_file)
    results = {}

    # Test 1: Save 5 fake labels and reload
    for item in test_labels:
        key = f"{item['clip']}_{item['frame']}"
        labels_data[key] = item
    save_labels_atomic(test_file, labels_data)

    reloaded = load_labels(test_file)
    all_match = True
    for item in test_labels:
        key = f"{item['clip']}_{item['frame']}"
        if key not in reloaded or reloaded[key] != item:
            all_match = False
            break
    results["save_5_fake_labels"] = "PASS" if all_match else "FAIL"

    # Test 2: Overwrite the same frame twice
    item_v1 = test_labels[0].copy()
    item_v1["h1_obj"] = "first_version"
    key_ov = f"{item_v1['clip']}_{item_v1['frame']}"
    labels_data[key_ov] = item_v1
    save_labels_atomic(test_file, labels_data)

    item_v2 = test_labels[0].copy()
    item_v2["h1_obj"] = "second_version_overwritten"
    labels_data[key_ov] = item_v2
    save_labels_atomic(test_file, labels_data)

    reloaded_ov = load_labels(test_file)
    results["overwrite_same_frame"] = "PASS" if reloaded_ov[key_ov]["h1_obj"] == "second_version_overwritten" else "FAIL"

    # Test 3: Save two different frames in a row
    item_a = {"clip": "seq_clip", "frame": 1, "t": 0.033, "h1_found": "No hand", "h1_role": "None", "h1_side": "None", "h1_kp_pass": "Fail", "h1_contact": "none", "h1_obj": "NO_HAND", "h2_found": "No hand", "h2_present": False, "h2_role": "None", "h2_side": "None", "h2_kp_pass": "Fail", "h2_contact": "none", "h2_obj": "NO_HAND", "source": "human_verified"}
    item_b = {"clip": "seq_clip", "frame": 2, "t": 0.067, "h1_found": "No hand", "h1_role": "None", "h1_side": "None", "h1_kp_pass": "Fail", "h1_contact": "none", "h1_obj": "NO_HAND", "h2_found": "No hand", "h2_present": False, "h2_role": "None", "h2_side": "None", "h2_kp_pass": "Fail", "h2_contact": "none", "h2_obj": "NO_HAND", "source": "human_verified"}
    
    labels_data["seq_clip_1"] = item_a
    save_labels_atomic(test_file, labels_data)
    labels_data["seq_clip_2"] = item_b
    save_labels_atomic(test_file, labels_data)

    reloaded_seq = load_labels(test_file)
    results["save_two_consecutive_frames"] = "PASS" if ("seq_clip_1" in reloaded_seq and "seq_clip_2" in reloaded_seq) else "FAIL"

    # Cleanup temp directory
    shutil.rmtree(test_dir)

    print("Test Results Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    return all(v == "PASS" for v in results.values())

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

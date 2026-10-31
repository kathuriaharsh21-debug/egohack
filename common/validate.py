"""Validator for pipeline outputs against schema specifications."""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List


def validate_hands_jsonl(jsonl_path: Path) -> Tuple[bool, List[str]]:
    """Validate hands.jsonl structure and value ranges."""
    errors = []
    if not jsonl_path.exists():
        return False, [f"File does not exist: {jsonl_path}"]

    line_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                errors.append(f"Line {idx+1}: Invalid JSON - {e}")
                if len(errors) > 20:
                    break
                continue

            for req in ["frame", "t", "hands"]:
                if req not in data:
                    errors.append(f"Line {idx+1}: Missing required field '{req}'")

            if not isinstance(data.get("frame"), int):
                errors.append(f"Line {idx+1}: 'frame' must be integer")

            if not isinstance(data.get("t"), (int, float)):
                errors.append(f"Line {idx+1}: 't' must be numeric")

            hands = data.get("hands", [])
            if not isinstance(hands, list):
                errors.append(f"Line {idx+1}: 'hands' must be a list")
                continue

            for h_idx, h in enumerate(hands):
                hand_reqs = ["id", "side", "bbox", "score", "kp2d", "kp3d", "interp", "visible"]
                for req in hand_reqs:
                    if req not in h:
                        errors.append(f"Line {idx+1}, hand {h_idx}: Missing '{req}'")

                if h.get("side") not in ["left", "right"]:
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'side' must be 'left' or 'right', got {h.get('side')}")

                bbox = h.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'bbox' must be 4 floats [xmin, ymin, xmax, ymax]")

                score = h.get("score")
                if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'score' must be in [0.0, 1.0]")

                kp2d = h.get("kp2d", [])
                if not isinstance(kp2d, list) or len(kp2d) != 21:
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'kp2d' must have 21 keypoints")

                kp3d = h.get("kp3d", [])
                if not isinstance(kp3d, list) or len(kp3d) != 21:
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'kp3d' must have 21 keypoints")

                if not isinstance(h.get("interp"), bool):
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'interp' must be boolean")

                if not isinstance(h.get("visible"), bool):
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'visible' must be boolean")

                if "is_wearer" in h and not isinstance(h.get("is_wearer"), bool):
                    errors.append(f"Line {idx+1}, hand {h_idx}: 'is_wearer' must be boolean")

            if len(errors) > 20:
                break

    if line_count == 0:
        errors.append("File is empty")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        valid, errs = validate_hands_jsonl(path)
        if valid:
            print(f"Validation PASSED for {path}")
            sys.exit(0)
        else:
            print(f"Validation FAILED for {path}:")
            for err in errs:
                print(f"  - {err}")
            sys.exit(1)
    else:
        print("Usage: python validate.py <path_to_hands.jsonl>")

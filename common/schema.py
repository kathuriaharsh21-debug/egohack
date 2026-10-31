"""Schema definitions for EgoHack outputs.

Defines the structure for hands.jsonl, objects.jsonl, contacts_frames.jsonl,
contacts.csv, tool_usage.csv, review_queue.csv, and metrics_hands.json.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class HandDetection:
    id: int                     # Tracking ID for hand instance
    side: str                   # 'left' or 'right'
    bbox: List[float]           # [x_min, y_min, x_max, y_max] in 480p pixel coordinates
    score: float                # Detection/presence confidence [0.0, 1.0]
    kp2d: List[List[float]]     # 21 landmarks: [[x, y], ...] in 480p pixels
    kp3d: List[List[float]]     # 21 landmarks: [[x, y, z], ...] in camera/world coordinates (meters)
    interp: bool                # True if interpolated/propagated, False if detected
    visible: bool               # True if detected, False if predicted (never counted as detection)
    is_wearer: bool = True      # True if wearer's hand, False if bystander / other person

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "side": str(self.side),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "score": round(float(self.score), 4),
            "kp2d": [[round(float(c), 2) for c in pt] for pt in self.kp2d],
            "kp3d": [[round(float(c), 4) for c in pt] for pt in self.kp3d],
            "interp": bool(self.interp),
            "visible": bool(self.visible),
            "is_wearer": bool(self.is_wearer),
        }


@dataclass
class FrameHands:
    frame: int                  # Original frame index (0-indexed)
    t: float                    # Timestamp in seconds
    hands: List[HandDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame),
            "t": round(float(self.t), 4),
            "hands": [h.to_dict() if isinstance(h, HandDetection) else h for h in self.hands]
        }

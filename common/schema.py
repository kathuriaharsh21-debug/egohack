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


@dataclass
class ObjectDetection:
    id: int
    label: str                  # Object class name
    bbox: List[float]           # [x_min, y_min, x_max, y_max] in 480p coordinates
    score: float                # Confidence score [0.0, 1.0]
    class_type: str = "portable"  # 'portable' or 'fixed'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "label": str(self.label),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "score": round(float(self.score), 4),
            "class_type": str(self.class_type),
        }


@dataclass
class FrameObjects:
    frame: int
    t: float
    objects: List[ObjectDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame),
            "t": round(float(self.t), 4),
            "objects": [o.to_dict() if isinstance(o, ObjectDetection) else o for o in self.objects]
        }


@dataclass
class HandContact:
    hand_id: int
    side: str                   # 'left' or 'right'
    state: str                  # 'none', 'portable', 'fixed'
    object_name: str            # Object label or 'none'
    object_id: Optional[int]    # Object tracking ID or None
    confidence: float           # Confidence score [0.0, 1.0]
    held_frames: int = 1        # Consecutive frames in contact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_id": int(self.hand_id),
            "side": str(self.side),
            "state": str(self.state),
            "object_name": str(self.object_name),
            "object_id": int(self.object_id) if self.object_id is not None else None,
            "confidence": round(float(self.confidence), 4),
            "held_frames": int(self.held_frames),
        }


@dataclass
class FrameContacts:
    frame: int
    t: float
    contacts: List[HandContact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame),
            "t": round(float(self.t), 4),
            "contacts": [c.to_dict() if isinstance(c, HandContact) else c for c in self.contacts]
        }

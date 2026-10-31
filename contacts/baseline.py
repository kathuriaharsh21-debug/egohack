"""PS2 Contact State Estimator Baseline.

Evaluates hand-object contact per frame:
- Only hands with visible=True AND is_wearer=True are eligible.
- Checks fingertips [4, 8, 12, 16, 20].
- Contact triggered if >= 2 fingertips inside object bbox expanded by 15% of hand size.
- Confirmed only if held for >= 3 consecutive frames.
- Classifies contact state as 'fixed' or 'portable' based on class-to-type map.
"""

from typing import List, Dict, Any, Optional, Set
import math
from common.schema import HandDetection, ObjectDetection, HandContact, FrameContacts

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
DEFAULT_FIXED_CLASSES = {"conveyor", "table", "shelf", "machine"}


class ContactEstimator:
    def __init__(
        self,
        fixed_classes: Optional[Set[str]] = None,
        expansion_ratio: float = 0.15,
        min_fingertips: int = 2,
        min_held_frames: int = 3,
    ):
        self.fixed_classes = fixed_classes or DEFAULT_FIXED_CLASSES
        self.expansion_ratio = expansion_ratio
        self.min_fingertips = min_fingertips
        self.min_held_frames = min_held_frames

        # Tracking state per hand_id -> {candidate_label, candidate_obj_id, held_count, confirmed_state, score}
        self.hand_history: Dict[int, Dict[str, Any]] = {}

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        wearer_hands: List[HandDetection],
        detected_objects: List[ObjectDetection],
    ) -> FrameContacts:
        """Estimate contact for all hands in the current frame."""
        frame_contacts: List[HandContact] = []
        active_hand_ids = set()

        for hand in wearer_hands:
            active_hand_ids.add(hand.id)

            # Rule: ONLY hands with visible=true AND is_wearer=true.
            # visible=false hands and bystanders never create contact.
            if not hand.visible or not hand.is_wearer:
                frame_contacts.append(
                    HandContact(
                        hand_id=hand.id,
                        side=hand.side,
                        state="none",
                        object_name="none",
                        object_id=None,
                        confidence=0.0,
                        held_frames=0,
                    )
                )
                continue

            # Compute hand size (max dimension of bbox)
            hw = max(0.0, hand.bbox[2] - hand.bbox[0])
            hh = max(0.0, hand.bbox[3] - hand.bbox[1])
            hand_size = max(hw, hh, 10.0)
            margin = self.expansion_ratio * hand_size

            # Extract 5 fingertips
            fingertip_pts = []
            for idx in FINGERTIP_INDICES:
                if idx < len(hand.kp2d):
                    fingertip_pts.append(hand.kp2d[idx])

            # Check candidate objects
            best_obj: Optional[ObjectDetection] = None
            max_in_count = 0

            for obj in detected_objects:
                # Expand object box by 15% of hand size
                exp_box = [
                    obj.bbox[0] - margin,
                    obj.bbox[1] - margin,
                    obj.bbox[2] + margin,
                    obj.bbox[3] + margin,
                ]

                # Count fingertips inside
                inside_count = 0
                for pt in fingertip_pts:
                    if exp_box[0] <= pt[0] <= exp_box[2] and exp_box[1] <= pt[1] <= exp_box[3]:
                        inside_count += 1

                if inside_count >= self.min_fingertips:
                    if inside_count > max_in_count:
                        max_in_count = inside_count
                        best_obj = obj
                    elif inside_count == max_in_count and best_obj is not None:
                        # Tie-break by object confidence
                        if obj.score > best_obj.score:
                            best_obj = obj

            # Update temporal confirmation history
            prev = self.hand_history.get(hand.id)

            if best_obj is not None:
                if prev is not None and prev["label"] == best_obj.label and (prev["obj_id"] == best_obj.id or prev["obj_id"] is None):
                    held = prev["held"] + 1
                else:
                    held = 1

                self.hand_history[hand.id] = {
                    "label": best_obj.label,
                    "obj_id": best_obj.id,
                    "held": held,
                    "score": best_obj.score,
                }

                if held >= self.min_held_frames:
                    state = "fixed" if best_obj.label in self.fixed_classes else "portable"
                    contact = HandContact(
                        hand_id=hand.id,
                        side=hand.side,
                        state=state,
                        object_name=best_obj.label,
                        object_id=best_obj.id,
                        confidence=best_obj.score,
                        held_frames=held,
                    )
                else:
                    contact = HandContact(
                        hand_id=hand.id,
                        side=hand.side,
                        state="none",
                        object_name="none",
                        object_id=None,
                        confidence=0.0,
                        held_frames=held,
                    )
            else:
                # No contact candidate in this frame
                self.hand_history[hand.id] = {
                    "label": "none",
                    "obj_id": None,
                    "held": 0,
                    "score": 0.0,
                }
                contact = HandContact(
                    hand_id=hand.id,
                    side=hand.side,
                    state="none",
                    object_name="none",
                    object_id=None,
                    confidence=0.0,
                    held_frames=0,
                )

            frame_contacts.append(contact)

        # Clean up stale hand histories
        stale_ids = [hid for hid in self.hand_history if hid not in active_hand_ids]
        for hid in stale_ids:
            del self.hand_history[hid]

        return FrameContacts(
            frame=frame_idx,
            t=timestamp_sec,
            contacts=frame_contacts,
        )

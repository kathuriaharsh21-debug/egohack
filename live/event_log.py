"""Event log: tracks GRAB / RELEASE per hand and writes a CSV.

Rules:
- A GRAB event starts when contact state changes from 'none' to something else.
- A RELEASE event fires when it goes back to 'none'.
- Confidence = mean object score during the held span * (held_frames / span_frames).
- Events with confidence < LOW_CONF_THRESHOLD go to the "needs human review" list.
- CSV is written to live/recordings/events_<timestamp>.csv.
"""

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from common.schema import FrameContacts, HandContact

LOW_CONF_THRESHOLD = 0.6


@dataclass
class ContactEvent:
    hand_id: int
    side: str
    object_name: str
    state: str          # 'portable' | 'fixed'
    t_start: float
    t_end: Optional[float] = None
    confidence: float = 0.0
    needs_review: bool = False
    # Accumulated during hold
    _scores: List[float] = field(default_factory=list, repr=False)

    def finalize(self, t_end: float) -> None:
        self.t_end = t_end
        raw_conf = float(sum(self._scores) / max(1, len(self._scores)))
        self.confidence = round(raw_conf, 3)
        self.needs_review = self.confidence < LOW_CONF_THRESHOLD

    def to_dict(self) -> dict:
        t_s = round(float(self.t_start), 3)
        t_e = round(float(self.t_end if self.t_end is not None else self.t_start), 3)
        duration = round(max(0.0, t_e - t_s), 3)
        return {
            "object_name": self.object_name,
            "state": self.state,
            "hand_side": self.side,
            "hand_id": int(self.hand_id),
            "t_start": t_s,
            "t_end": t_e,
            "duration_sec": duration,
            "confidence": float(self.confidence),
            "needs_review": bool(self.needs_review),
        }


class EventLog:
    """Accumulates contact events and persists them to both CSV and JSON."""

    def __init__(self, recordings_dir: str = "live/recordings") -> None:
        os.makedirs(recordings_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recordings_dir = recordings_dir
        self.csv_path = os.path.join(recordings_dir, f"events_{ts}.csv")
        self.json_path = os.path.join(recordings_dir, f"events_{ts}.json")
        self.latest_json_path = os.path.join(recordings_dir, "events_latest.json")
        self._init_csv()
        self._write_json()

        # State per hand_id
        self._active: Dict[int, ContactEvent] = {}

        # Completed events (all + review-flagged)
        self.events: List[ContactEvent] = []
        self.review_events: List[ContactEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_contacts: FrameContacts) -> None:
        """Ingest one frame of contact data."""
        ts = frame_contacts.t if hasattr(frame_contacts, "t") else 0.0
        seen_ids = set()

        for hc in frame_contacts.contacts:
            seen_ids.add(hc.hand_id)
            if hc.state == "none":
                self._close_event(hc.hand_id, ts)
            else:
                # Active contact
                if hc.hand_id not in self._active:
                    # New GRAB event
                    ev = ContactEvent(
                        hand_id=hc.hand_id,
                        side=hc.side,
                        object_name=hc.object_name,
                        state=hc.state,
                        t_start=ts,
                    )
                    self._active[hc.hand_id] = ev
                else:
                    ev = self._active[hc.hand_id]
                    ev._scores.append(float(hc.confidence))

        # Close events for hands no longer in frame
        for hand_id in list(self._active):
            if hand_id not in seen_ids:
                self._close_event(hand_id, ts)

    def reset(self) -> None:
        """Clear all events (keyboard R)."""
        for hand_id, ev in list(self._active.items()):
            ev.finalize(t_end=0.0)
        self._active.clear()
        self.events.clear()
        self.review_events.clear()

    def get_display_events(self, n: int = 12) -> List[ContactEvent]:
        """Return the last n completed events for the panel."""
        return self.events[-n:]

    def get_review_events(self, n: int = 6) -> List[ContactEvent]:
        return self.review_events[-n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def finalize_all(self, final_ts: float) -> List[dict]:
        """Close any remaining active events on shutdown and flush JSON/CSV."""
        for hand_id in list(self._active):
            self._close_event(hand_id, final_ts)
        self._write_json()
        return [ev.to_dict() for ev in self.events]

    def _close_event(self, hand_id: int, ts: float) -> None:
        if hand_id not in self._active:
            return
        ev = self._active.pop(hand_id)
        ev.finalize(t_end=ts)
        self.events.append(ev)
        if ev.needs_review:
            self.review_events.append(ev)
        self._write_row(ev)
        self._write_json()
        # Print real-time JSON format to console
        print(f"[TOUCH EVENT JSON] {json.dumps(ev.to_dict())}")

    def _init_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "hand_id", "side", "object_name", "state",
                "t_start", "t_end", "confidence", "needs_review"
            ])

    def _write_row(self, ev: ContactEvent) -> None:
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ev.hand_id, ev.side, ev.object_name, ev.state,
                    round(ev.t_start, 3), round(ev.t_end or 0.0, 3),
                    ev.confidence, int(ev.needs_review)
                ])
        except Exception:
            pass

    def _write_json(self) -> None:
        try:
            records = [ev.to_dict() for ev in self.events]
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            with open(self.latest_json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass


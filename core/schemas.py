from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"class_name": self.class_name, "confidence": round(float(self.confidence), 6), "bbox": [round(float(x), 3) for x in self.bbox], "track_id": self.track_id}


@dataclass
class SafetyEvent:
    event_type: str
    status: str
    confidence: float
    source: str
    timestamp: str = field(default_factory=utc_now)
    image_path: str | None = None
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "event_type": self.event_type, "status": self.status, "confidence": round(float(self.confidence), 6), "source": self.source, "timestamp": self.timestamp, "image_path": self.image_path, "reason": self.reason, "payload": self.payload}

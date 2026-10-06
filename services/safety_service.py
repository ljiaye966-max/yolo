from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import monotonic

from core.schemas import Detection, SafetyEvent, utc_now


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def overlap_ratio(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    denominator = _area(inner)
    return _intersection(inner, outer) / denominator if denominator else 0.0


def center_in_zone(box: tuple[float, float, float, float], person: tuple[float, float, float, float], start: float, end: float) -> bool:
    width = person[2] - person[0]; height = person[3] - person[1]
    cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
    return person[0] - width * 0.15 <= cx <= person[2] + width * 0.15 and person[1] + height * start <= cy <= person[1] + height * end


@dataclass
class _TemporalState:
    count: int = 0
    last_emit: float = 0.0


class SafetyRuleEngine:
    """Convert detections into conservative, reviewable safety events."""

    def __init__(
        self,
        min_confidence: float = 0.35,
        min_consecutive_frames: int = 3,
        cooldown_seconds: float = 10.0,
        *,
        infer_missing_helmet_requires_vest: bool = False,
        missing_helmet_min_person_height_px: float = 0.0,
        helmet_upper_zone: float = 0.45,
        class_min_confidence: dict[str, float] | None = None,
    ):
        self.min_confidence = min_confidence
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        # A missing helmet prediction is ambiguous: it can mean either a real
        # violation or a detector miss.  Strict mode only infers a violation
        # when another PPE signal (vest) supports the person association.
        self.infer_missing_helmet_requires_vest = infer_missing_helmet_requires_vest
        self.missing_helmet_min_person_height_px = max(0.0, missing_helmet_min_person_height_px)
        self.helmet_upper_zone = min(1.0, max(0.1, helmet_upper_zone))
        self.class_min_confidence = class_min_confidence or {}
        self._states: dict[str, _TemporalState] = defaultdict(_TemporalState)

    def evaluate(self, detections: list[Detection], source: str, image_path: str | None = None, timestamp: str | None = None) -> list[SafetyEvent]:
        usable = [d for d in detections if d.confidence >= self.class_min_confidence.get(d.class_name, self.min_confidence)]
        persons = [d for d in usable if d.class_name == "person"]
        helmets = [d for d in usable if d.class_name == "helmet"]
        vests = [d for d in usable if d.class_name == "vest"]
        no_helmets = [d for d in usable if d.class_name == "no_helmet"]
        candidates: list[tuple[str, float, str, dict]] = []
        for index, person in enumerate(persons):
            person_height = max(0.0, person.bbox[3] - person.bbox[1])
            helmet = any(center_in_zone(h.bbox, person.bbox, 0.0, self.helmet_upper_zone) or overlap_ratio(h.bbox, person.bbox) > 0.10 for h in helmets)
            vest = any(center_in_zone(v.bbox, person.bbox, 0.25, 0.95) or overlap_ratio(v.bbox, person.bbox) > 0.10 for v in vests)
            explicit_no_helmet = any(overlap_ratio(n.bbox, person.bbox) > 0.05 or center_in_zone(n.bbox, person.bbox, 0.0, self.helmet_upper_zone) for n in no_helmets)
            inferred_missing_helmet = (
                not helmet
                and (not self.infer_missing_helmet_requires_vest or vest)
                and person_height >= self.missing_helmet_min_person_height_px
            )
            if explicit_no_helmet or inferred_missing_helmet:
                reason = "原始模型标注/检测到疑似缺少安全帽" if explicit_no_helmet else "人员区域内未匹配到安全帽，可能受遮挡、视角或漏检影响"
                confidence = max(0.0, min(1.0, person.confidence * (0.95 if explicit_no_helmet else 0.65)))
                candidates.append((f"person-{index}", confidence, reason, {
                    "person": person.as_dict(),
                    "helmet_matched": helmet,
                    "vest_matched": vest,
                    "explicit_no_helmet": explicit_no_helmet,
                    "inferred_missing_helmet": inferred_missing_helmet,
                }))
        events: list[SafetyEvent] = []
        for key, confidence, reason, payload in candidates:
            state = self._states[key]; state.count += 1; now = monotonic()
            if state.count < self.min_consecutive_frames or now - state.last_emit < self.cooldown_seconds: continue
            state.last_emit = now
            events.append(SafetyEvent(event_type="suspected_violation", status="needs_human_review", confidence=confidence, source=source, timestamp=timestamp or utc_now(), image_path=image_path, reason=reason, payload=payload))
        active = {item[0] for item in candidates}
        for key, state in self._states.items():
            if key not in active: state.count = 0
        return events

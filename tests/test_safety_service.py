from core.schemas import Detection
from services.safety_service import SafetyRuleEngine, overlap_ratio


def test_overlap_ratio():
    assert overlap_ratio((1, 1, 3, 3), (0, 0, 4, 4)) == 1.0


def test_missing_helmet_requires_consecutive_frames():
    engine = SafetyRuleEngine(min_confidence=0.2, min_consecutive_frames=2, cooldown_seconds=0); detections = [Detection("person", 0.9, (0, 0, 100, 200))]
    assert engine.evaluate(detections, "test") == []; events = engine.evaluate(detections, "test"); assert len(events) == 1; assert events[0].event_type == "suspected_violation"; assert events[0].status == "needs_human_review"


def test_explicit_no_helmet_is_still_reviewable():
    engine = SafetyRuleEngine(min_confidence=0.2, min_consecutive_frames=1, cooldown_seconds=0); detections = [Detection("person", 0.9, (0, 0, 100, 200)), Detection("no_helmet", 0.8, (20, 0, 80, 60))]; events = engine.evaluate(detections, "test"); assert events[0].status == "needs_human_review"; assert events[0].payload["explicit_no_helmet"] is True


def test_strict_mode_does_not_turn_helmet_miss_into_violation_without_vest():
    engine = SafetyRuleEngine(min_confidence=0.2, min_consecutive_frames=1, cooldown_seconds=0, infer_missing_helmet_requires_vest=True, missing_helmet_min_person_height_px=80)
    detections = [Detection("person", 0.9, (0, 0, 100, 200))]
    assert engine.evaluate(detections, "test") == []


def test_strict_mode_keeps_missing_helmet_with_vest_reviewable():
    engine = SafetyRuleEngine(min_confidence=0.2, min_consecutive_frames=1, cooldown_seconds=0, infer_missing_helmet_requires_vest=True, missing_helmet_min_person_height_px=80)
    detections = [Detection("person", 0.9, (0, 0, 100, 200)), Detection("vest", 0.8, (20, 60, 80, 170))]
    events = engine.evaluate(detections, "test")
    assert len(events) == 1
    assert events[0].payload["inferred_missing_helmet"] is True

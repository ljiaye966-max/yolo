from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from core.schemas import Detection
from services.database_service import EventStore
from services.inference_service import InferenceService
from services.safety_service import SafetyRuleEngine


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--weights"); parser.add_argument("--save-dir", default="runs/inference"); args = parser.parse_args(); settings = get_settings(); source = Path(args.source).resolve()
    if not source.exists(): raise SystemExit(f"source not found: {source}")
    service = InferenceService(
        Path(args.weights).resolve() if args.weights else settings.model_path,
        settings.model.get("conf", 0.10),
        settings.model.get("iou", 0.45),
        settings.model.get("class_confidence", {}),
        int(settings.model.get("inference_imgsz", 960)),
    ); raw_results = service.predict(source, Path(args.save_dir) / f"{source.stem}_annotated.jpg")
    engine = SafetyRuleEngine(
        settings.rules.get("min_confidence", 0.35),
        settings.rules.get("min_consecutive_frames", 3),
        settings.rules.get("cooldown_seconds", 10),
        infer_missing_helmet_requires_vest=settings.rules.get("infer_missing_helmet_requires_vest", True),
        missing_helmet_min_person_height_px=settings.rules.get("missing_helmet_min_person_height_px", 80),
        helmet_upper_zone=settings.rules.get("helmet_upper_zone", 0.45),
        class_min_confidence=settings.model.get("class_confidence", {}),
    )
    store = EventStore(settings.database_path); events = []
    for result in raw_results:
        detections = [Detection(d["class_name"], d["confidence"], tuple(d["bbox"])) for d in result["detections"]]
        for event in engine.evaluate(detections, str(source), result.get("annotated_path")): event.id = store.add(event); events.append(event.as_dict())
    print(json.dumps({"results": raw_results, "events": events}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

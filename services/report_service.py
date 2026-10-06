from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings
from services.database_service import EventStore


def generate_report(settings: Settings, metrics_path: Path | None = None) -> Path:
    settings.ensure_directories(); store = EventStore(settings.database_path); metrics = {}
    if metrics_path and metrics_path.exists(): metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "scope": "teaching prototype; suspected events require human review", "event_summary": store.count_by_type(), "events": store.list(10000), "model_metrics": metrics, "limitations": ["model errors and camera angle affect recall and precision", "events are not final safety determinations", "no person identity inference is performed"]}
    path = settings.project_root / "reports" / f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

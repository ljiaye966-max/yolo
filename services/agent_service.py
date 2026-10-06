from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.config import Settings, safe_project_path


@dataclass
class ToolResult:
    tool_name: str
    status: str
    summary: str
    data: dict[str, Any]


class SafeAgent:
    """Allow-listed tool facade; intentionally not an autonomous executor."""

    def __init__(self, settings: Settings):
        self.settings = settings; self.tools: dict[str, Callable[..., ToolResult]] = {"validate_dataset": self.validate_dataset, "convert_dataset": self.convert_dataset, "evaluate_model": self.evaluate_model, "run_inference": self.run_inference, "query_events": self.query_events, "generate_report": self.generate_report, "train_model": self.train_model}

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self.tools: return ToolResult(name, "rejected", "tool is not allow-listed", {})
        try: return self.tools[name](**kwargs)
        except Exception as exc: return ToolResult(name, "failed", f"{type(exc).__name__}: {exc}", {})

    def validate_dataset(self, **_: Any) -> ToolResult:
        from services.validation_service import validate_dataset
        return ToolResult("validate_dataset", "ok", "dataset validation completed", validate_dataset(self.settings))

    def convert_dataset(self, **_: Any) -> ToolResult:
        from services.dataset_service import convert_construction_ppe
        return ToolResult("convert_dataset", "ok", "dataset conversion completed", convert_construction_ppe(self.settings))

    def evaluate_model(self, **_: Any) -> ToolResult: return ToolResult("evaluate_model", "not_implemented", "run scripts/evaluate_model.py with a trained checkpoint", {})

    def run_inference(self, source: str, **_: Any) -> ToolResult:
        safe_source = safe_project_path(source); from services.inference_service import InferenceService
        result = InferenceService(self.settings.model_path, float(self.settings.model.get("conf", 0.35)), float(self.settings.model.get("iou", 0.45))).predict(safe_source); return ToolResult("run_inference", "ok", "inference completed", {"results": result})

    def query_events(self, limit: int = 100, **_: Any) -> ToolResult:
        from services.database_service import EventStore
        return ToolResult("query_events", "ok", "events queried", {"events": EventStore(self.settings.database_path).list(limit)})

    def generate_report(self, **_: Any) -> ToolResult:
        from services.report_service import generate_report
        path = generate_report(self.settings); return ToolResult("generate_report", "ok", "report generated", {"path": str(path)})

    def train_model(self, **_: Any) -> ToolResult: return ToolResult("train_model", "confirmation_required", "formal training needs explicit user confirmation and is not run by the agent", {})

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings(raw)


class Settings:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.project_root = PROJECT_ROOT
        self.classes = list(raw.get("classes", ["person", "helmet", "vest", "no_helmet"]))
        self.dataset = raw.get("dataset", {})
        self.model = raw.get("model", {})
        self.rules = raw.get("rules", {})
        self.database = raw.get("database", {})
        self.llm = raw.get("llm", {})

    @property
    def raw_dir(self) -> Path:
        return _resolve(self.dataset.get("raw_dir", "datasets/raw"))

    @property
    def processed_dir(self) -> Path:
        return _resolve(self.dataset.get("processed_dir", "datasets/processed"))

    @property
    def report_dir(self) -> Path:
        return _resolve(self.dataset.get("report_dir", "datasets/reports/data_quality"))

    @property
    def quarantine_dir(self) -> Path:
        return _resolve(self.dataset.get("quarantine_dir", "datasets/quarantine"))

    @property
    def database_path(self) -> Path:
        return _resolve(self.database.get("path", "data/events.sqlite3"))

    @property
    def model_path(self) -> Path:
        return _resolve(self.model.get("weights", "yolo11n.pt"))

    @property
    def data_yaml_path(self) -> Path:
        return _resolve(self.model.get("data", "datasets/processed/data.yaml"))

    @property
    def runs_dir(self) -> Path:
        return _resolve(self.model.get("runs_dir", "runs"))

    def ensure_directories(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.report_dir, self.quarantine_dir, self.database_path.parent, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)


def safe_project_path(candidate: str | Path, *, allow_raw_write: bool = False) -> Path:
    """Resolve a user path while preventing traversal outside the project."""

    root = PROJECT_ROOT.resolve()
    path = Path(candidate)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path must stay inside the project directory")
    raw_dir = get_settings().raw_dir.resolve()
    if not allow_raw_write and (resolved == raw_dir or raw_dir in resolved.parents):
        raise ValueError("raw data is read-only for conversion and agent tools")
    return resolved

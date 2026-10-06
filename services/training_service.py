from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Settings


class TrainService:
    def __init__(self, settings: Settings): self.settings = settings

    def train(self, *, epochs: int | None = None, batch: int | None = None, imgsz: int | None = None, lr0: float | None = None, device: str | None = None, workers: int | None = None, resume: str | None = None, weights: str | None = None, data: str | None = None, confirm: bool = False, smoke_test: bool = False) -> Path:
        if not confirm: raise PermissionError("formal training requires --confirm after reviewing model, dataset, epochs, device and output directory")
        data_path = Path(data) if data else self.settings.data_yaml_path
        if not data_path.is_absolute(): data_path = self.settings.project_root / data_path
        data_path = data_path.resolve()
        if not data_path.exists(): raise FileNotFoundError(f"dataset yaml not found: {data_path}; run conversion and validation first")
        # Windows installations that combine PyTorch, NumPy and OpenCV can load
        # two Intel OpenMP runtimes. Keep the workaround scoped to training.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # The bundled Polars wheel advertises an sse3 flag that this build's
        # CPU feature parser does not recognize; bypass only that parser.
        os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
        try: from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("ultralytics is not installed in the selected interpreter") from exc
        run_name = datetime.now().strftime("train_%Y%m%d_%H%M%S_%f"); run_dir = self.settings.runs_dir / run_name
        initial_weights = weights or str(self.settings.model.get("weights", "yolo11n.pt"))
        weights_path = Path(initial_weights)
        if not weights_path.is_absolute(): weights_path = self.settings.project_root / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.exists() and not resume: raise FileNotFoundError(f"model weights not found: {weights_path}")
        args: dict[str, Any] = {"data": str(data_path), "epochs": epochs or self.settings.model.get("epochs", 100), "batch": batch or self.settings.model.get("batch", 16), "imgsz": imgsz or self.settings.model.get("imgsz", 640), "lr0": lr0 or self.settings.model.get("lr0", 0.01), "device": device or self.settings.model.get("device", "auto"), "workers": workers if workers is not None else self.settings.model.get("workers", 0), "amp": bool(self.settings.model.get("amp", False)), "plots": bool(self.settings.model.get("plots", False)), "cache": bool(self.settings.model.get("cache", False)), "seed": self.settings.model.get("seed", 42), "project": str(self.settings.runs_dir), "name": run_name, "exist_ok": False}
        if smoke_test: args.update({"epochs": 1, "batch": 2, "workers": 0, "device": "cpu", "amp": False, "plots": False, "cache": False, "fraction": 0.02, "val": False})
        metadata = self._environment_metadata(args, resume)
        model = YOLO(resume or str(weights_path))
        try:
            result = model.train(resume=bool(resume), **args) if resume else model.train(**args); results = getattr(result, "results_dict", {}) or {}; run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); (run_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); (run_dir / "failure.log").write_text(str(exc), encoding="utf-8"); raise
        return run_dir

    @staticmethod
    def _environment_metadata(args: dict[str, Any], resume: str | None) -> dict:
        metadata = {"python": sys.version, "platform": platform.platform(), "args": args, "resume": resume, "kmp_duplicate_lib_ok": os.getenv("KMP_DUPLICATE_LIB_OK"), "polars_skip_cpu_check": os.getenv("POLARS_SKIP_CPU_CHECK")}
        try:
            import torch
            metadata.update({"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        except ImportError: metadata["torch"] = None
        try:
            import ultralytics
            metadata["ultralytics"] = ultralytics.__version__
        except ImportError: metadata["ultralytics"] = None
        return metadata

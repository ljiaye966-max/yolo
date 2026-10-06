from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--weights", required=True); parser.add_argument("--split", choices=["val", "test"], default="test"); args = parser.parse_args(); settings = get_settings()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
    try: from ultralytics import YOLO
    except ImportError as exc: raise SystemExit("ultralytics is not installed") from exc
    weights = Path(args.weights).resolve()
    if not weights.exists(): raise SystemExit(f"weights not found: {weights}")
    metrics = YOLO(str(weights)).val(data=str(settings.data_yaml_path), split=args.split, plots=False, workers=0, project=str(settings.runs_dir), name=f"val_{args.split}"); result = getattr(metrics, "results_dict", {}) or {}; output = settings.runs_dir / f"evaluation_{args.split}.json"; output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); print(f"saved={output}"); return 0


if __name__ == "__main__": raise SystemExit(main())

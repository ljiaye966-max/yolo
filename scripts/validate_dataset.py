import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from services.validation_service import validate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--quarantine", action="store_true"); parser.add_argument("--root"); args = parser.parse_args(); root = Path(args.root).resolve() if args.root else None; report = validate_dataset(get_settings(), quarantine=args.quarantine, root=root); print(json.dumps(report, ensure_ascii=False, indent=2)); raise SystemExit(1 if report["issue_count"] else 0)

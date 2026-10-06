from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from services.training_service import TrainService


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int); parser.add_argument("--batch", type=int); parser.add_argument("--imgsz", type=int); parser.add_argument("--lr0", type=float); parser.add_argument("--device"); parser.add_argument("--workers", type=int); parser.add_argument("--resume"); parser.add_argument("--weights"); parser.add_argument("--data"); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--confirm", action="store_true"); args = parser.parse_args(); settings = get_settings(); settings.ensure_directories()
    print(f"model={args.weights or settings.model.get('weights')}"); print(f"data={args.data or settings.data_yaml_path}"); print(f"epochs={1 if args.smoke_test else args.epochs or settings.model.get('epochs')}"); print(f"device={'cpu' if args.smoke_test else args.device or settings.model.get('device')}")
    run_dir = TrainService(settings).train(**vars(args)); print(f"run={run_dir}"); return 0


if __name__ == "__main__": raise SystemExit(main())

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from services.report_service import generate_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--metrics"); args = parser.parse_args(); print(generate_report(get_settings(), Path(args.metrics) if args.metrics else None))

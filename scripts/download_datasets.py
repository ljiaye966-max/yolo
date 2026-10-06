from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from services.dataset_service import OFFICIAL_SOURCES, download_construction_ppe, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets without deleting raw data"); parser.add_argument("--dataset", choices=["construction_ppe", "all"], default="construction_ppe"); parser.add_argument("--force", action="store_true"); args = parser.parse_args(); settings = get_settings(); settings.ensure_directories(); downloaded: list[str] = []
    if args.dataset in {"construction_ppe", "all"}:
        path = download_construction_ppe(settings, force=args.force); downloaded.append(f"construction_ppe:{path}"); print(f"ready: {path}")
    manifest = write_manifest(settings, downloaded); print(f"manifest: {manifest}")
    if args.dataset == "all":
        print("Supplementary sources are recorded in the manifest. SHWD/CHV direct files are hosted by external drives; verify terms before supplying a URL to a downloader.")
        for key, value in OFFICIAL_SOURCES.items(): print(f"{key}: {value['url']}")
    return 0


if __name__ == "__main__": raise SystemExit(main())

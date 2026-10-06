"""Entry point for the construction-site PPE monitoring prototype."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construction PPE monitoring system")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("web", help="Start the Streamlit dashboard")
    sub.add_parser("doctor", help="Check the local environment and project files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "web":
        from web.app import main as web_main

        return web_main()
    if args.command == "doctor":
        from core.config import get_settings

        settings = get_settings()
        print(f"project_root={settings.project_root}")
        print(f"database={settings.database_path}")
        print(f"model={settings.model_path}")
        print("ok")
        return 0
    print("Use `python main.py web` or one of the scripts in scripts/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

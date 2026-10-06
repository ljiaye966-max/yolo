from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from core.config import Settings


def validate_dataset(settings: Settings, quarantine: bool = False, root: Path | None = None) -> dict:
    root = root or settings.processed_dir; issues: list[dict] = []; duplicate_hashes: dict[str, Path] = {}; summary = {"splits": {}, "classes": Counter(), "issues": issues}; extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split; label_dir = root / "labels" / split; images = [p for p in image_dir.rglob("*") if p.suffix.lower() in extensions] if image_dir.exists() else []; split_info = {"images": len(images), "labels": 0, "empty_labels": 0}
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in duplicate_hashes: issues.append({"type": "duplicate_image", "path": str(image), "same_as": str(duplicate_hashes[digest])})
            else: duplicate_hashes[digest] = image
            label = label_dir / f"{image.stem}.txt"
            if not label.exists(): issues.append({"type": "missing_label", "path": str(image), "expected": str(label)}); continue
            split_info["labels"] += 1; content = label.read_text(encoding="utf-8", errors="ignore").strip()
            if not content: split_info["empty_labels"] += 1
            for line_no, line in enumerate(content.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5: issues.append({"type": "malformed_label", "path": str(label), "line": line_no}); continue
                try: class_id = int(parts[0]); values = [float(v) for v in parts[1:]]
                except ValueError: issues.append({"type": "non_numeric_label", "path": str(label), "line": line_no}); continue
                if not 0 <= class_id < 4 or any(v < 0 or v > 1 for v in values): issues.append({"type": "invalid_bbox", "path": str(label), "line": line_no, "values": values})
                summary["classes"][str(class_id)] += 1
        summary["splits"][split] = split_info
    report = {"splits": summary["splits"], "classes": dict(summary["classes"]), "issue_count": len(issues), "issues": issues}; settings.report_dir.mkdir(parents=True, exist_ok=True); (settings.report_dir / "dataset_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); _write_html(settings.report_dir / "dataset_quality.html", report)
    if quarantine and issues:
        settings.quarantine_dir.mkdir(parents=True, exist_ok=True)
        for issue in issues:
            source = Path(issue["path"])
            if source.exists() and source.is_file(): source.rename(settings.quarantine_dir / source.name)
    return report


def _write_html(path: Path, report: dict) -> None:
    rows = "".join(f"<tr><td>{i}</td><td>{html.escape(json.dumps(item, ensure_ascii=False))}</td></tr>" for i, item in enumerate(report["issues"], 1)); body = f"<!doctype html><meta charset='utf-8'><title>Dataset quality report</title><h1>Dataset quality report</h1><p>Issues: {report['issue_count']}</p><pre>{html.escape(json.dumps({'splits': report['splits'], 'classes': report['classes']}, ensure_ascii=False, indent=2))}</pre><table border='1'><tr><th>#</th><th>Issue</th></tr>{rows}</table>"; path.write_text(body, encoding="utf-8")

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from core.config import Settings


OFFICIAL_SOURCES = {
    "construction_ppe": {"url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip", "license": "AGPL-3.0", "format": "YOLO", "notes": "Official Ultralytics source; map only person/helmet/vest/no_helmet for this project."},
    "shwd": {"url": "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset", "download_url": "https://drive.google.com/file/d/1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX/view?usp=drive_open", "license": "MIT (verify original dataset terms)", "format": "Pascal VOC XML", "notes": "Repository metadata only; direct archive links are hosted by Baidu/Google Drive."},
    "voxel51_hard_hat_detection": {"url": "https://huggingface.co/datasets/Voxel51/hard-hat-detection", "license": "CC0-1.0 per dataset card", "format": "FiftyOne samples.json + images", "notes": "Head labels are mapped to no_helmet; absence of vest is not inferred."},
    "hardhat_workers": {"url": "https://public.roboflow.com/object-detection/hard-hat-workers", "license": "Public Domain per dataset page", "format": "YOLO/VOC", "notes": "Roboflow public export endpoint may require an account; not used when it returns an HTML page."},
    "pictor": {"url": "https://github.com/ciber-lab/pictor-ppe", "download_url": "https://drive.google.com/drive/folders/1akhyTNVrkqMMcIFUQCEbW5ehfmG0CdYH?usp=sharing", "license": "See upstream dataset/repository terms", "format": "upstream-specific", "notes": "Check the upstream dataset download and research-use terms."},
    "chv": {"url": "https://github.com/ZijianWang-ZW/PPE_detection", "download_url": "https://drive.google.com/file/d/1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a/view?usp=sharing", "license": "Free use per upstream README; verify before redistribution", "format": "upstream-specific", "notes": "Dataset files are linked from Google Drive/Baidu Yunpan, not stored in the repository."},
}


def download_construction_ppe(settings: Settings, force: bool = False) -> Path:
    settings.ensure_directories(); archive = settings.raw_dir / "construction-ppe.zip"; target = settings.raw_dir / "construction_ppe"
    if target.exists() and any(target.rglob("*.jpg")) and not force: return target
    if force and target.exists(): shutil.rmtree(target)
    if not archive.exists():
        print(f"Downloading Construction-PPE to {archive} ...")
        urllib.request.urlretrieve(settings.dataset.get("construction_ppe_url", OFFICIAL_SOURCES["construction_ppe"]["url"]), archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(settings.raw_dir)
    if not target.exists() and (settings.raw_dir / "images").exists() and (settings.raw_dir / "labels").exists():
        target.mkdir(parents=True, exist_ok=True)
        for item in ("images", "labels", "data.yaml", "dataset.yaml", "construction-ppe.yaml", "LICENSE"):
            source = settings.raw_dir / item
            if source.exists(): source.rename(target / item)
    candidates = [p for p in settings.raw_dir.iterdir() if p.is_dir() and (p / "images").exists() and (p / "labels").exists()]
    if candidates and candidates[0] != target: candidates[0].rename(target)
    if not target.exists(): raise FileNotFoundError("downloaded archive did not contain an images/labels dataset")
    return target


def write_manifest(settings: Settings, downloaded: list[str] | None = None) -> Path:
    settings.ensure_directories(); payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "project_classes": ["person", "helmet", "vest", "no_helmet"], "downloaded": downloaded or [], "sources": OFFICIAL_SOURCES, "policy": "No raw data deletion; unknown files are quarantined only by explicit validation command."}
    path = settings.raw_dir.parent / "manifest.json"; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); return path


def _find_dataset_root(raw_dir: Path) -> Path:
    direct = raw_dir / "construction_ppe"
    if (direct / "images").exists() and (direct / "labels").exists(): return direct
    for path in raw_dir.rglob("*"):
        if path.is_dir() and (path / "images").exists() and (path / "labels").exists(): return path
    raise FileNotFoundError("No dataset with images/ and labels/ found under datasets/raw")


def _read_names(root: Path) -> list[str]:
    for candidate in (root / "data.yaml", root / "dataset.yaml", root / "construction-ppe.yaml"):
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}; names = data.get("names", [])
            return [names[k] for k in sorted(names, key=lambda item: int(item))] if isinstance(names, dict) else list(names)
    return []


def _copy_split(root: Path, split: str, out_images: Path, out_labels: Path, names: list[str], source_dataset: str, manifest: dict[str, str]) -> Counter:
    counter = Counter(); image_dir = root / "images" / split; label_dir = root / "labels" / split
    if not image_dir.exists(): return counter
    for image in sorted(image_dir.rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}: continue
        relative_key = f"{source_dataset}/{split}/{image.relative_to(image_dir).as_posix()}"; stem = hashlib.sha1(relative_key.encode("utf-8")).hexdigest()[:16]
        target_image = out_images / f"{source_dataset}_{stem}{image.suffix.lower()}"; target_label = out_labels / f"{target_image.stem}.txt"; label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
        lines: list[str] = []
        if label.exists():
            for raw_line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = raw_line.split()
                if len(parts) != 5: continue
                try: index = int(parts[0]); values = [float(value) for value in parts[1:]]
                except ValueError: continue
                name = names[index] if 0 <= index < len(names) else ""; mapped = {"person": 0, "Person": 0, "helmet": 1, "vest": 2, "no_helmet": 3}.get(name)
                if mapped is None: continue
                lines.append(f"{mapped} " + " ".join(f"{value:.6f}" for value in values)); counter[mapped] += 1
        shutil.copy2(image, target_image); target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"); manifest[target_image.name] = source_dataset; counter["images"] += 1
    return counter


def convert_construction_ppe(settings: Settings, seed: int = 42) -> dict:
    settings.ensure_directories(); root = _find_dataset_root(settings.raw_dir); names = _read_names(root)
    if not names: raise ValueError(f"cannot find class names in {root}")
    for child in settings.processed_dir.iterdir() if settings.processed_dir.exists() else []:
        if child.name != "data.yaml" and child.is_dir(): shutil.rmtree(child)
    manifest: dict[str, str] = {}; summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        image_out = settings.processed_dir / "images" / split; label_out = settings.processed_dir / "labels" / split; image_out.mkdir(parents=True, exist_ok=True); label_out.mkdir(parents=True, exist_ok=True)
        summary[split] = dict(_copy_split(root, split, image_out, label_out, names, "construction_ppe", manifest))
    data = {"path": str(settings.processed_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {i: name for i, name in enumerate(["person", "helmet", "vest", "no_helmet"])} }
    (settings.processed_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"); (settings.processed_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"source_root": str(root), "source_names": names, "summary": summary, "class_mapping": {"Person": "person", "person": "person", "helmet": "helmet", "vest": "vest", "no_helmet": "no_helmet"}}
    settings.report_dir.mkdir(parents=True, exist_ok=True); (settings.report_dir / "conversion_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); return report

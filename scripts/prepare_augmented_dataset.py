from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PROJECT_ROOT, get_settings
from services.dataset_service import write_manifest


PROJECT_CLASSES = {"person": 0, "helmet": 1, "vest": 2, "no_helmet": 3}


def _copy_tree(source: Path, target: Path) -> dict[str, int]:
    counts = Counter()
    for split in ("train", "val", "test"):
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        target_images = target / "images" / split
        target_labels = target / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        if not source_images.exists():
            continue
        for image in source_images.iterdir():
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label = source_labels / f"{image.stem}.txt"
            target_name = f"construction_ppe_{image.name}"
            shutil.copy2(image, target_images / target_name)
            if label.exists():
                shutil.copy2(label, target_labels / f"{Path(target_name).stem}.txt")
            else:
                (target_labels / f"{Path(target_name).stem}.txt").write_text("", encoding="utf-8")
            counts[f"{split}_images"] += 1
    return dict(counts)


def _prepare_voxel51(raw_root: Path, target: Path, seed: int) -> dict:
    index_path = raw_root / "samples.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing Voxel51 index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))
    if not samples:
        raise ValueError("Voxel51 samples.json contains no samples")
    random.Random(seed).shuffle(samples)
    split_at = {"train": int(len(samples) * 0.8), "val": int(len(samples) * 0.9)}
    counts = Counter()
    class_counts = Counter()
    missing = []
    for index, sample in enumerate(samples):
        split = "train" if index < split_at["train"] else "val" if index < split_at["val"] else "test"
        relative = Path(sample["filepath"])
        image = raw_root / relative
        if not image.exists():
            missing.append(str(image))
            continue
        key = f"voxel51/{relative.as_posix()}"
        stem = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        target_image = target / "images" / split / f"voxel51_{stem}{image.suffix.lower()}"
        target_label = target / "labels" / split / f"{target_image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        lines = []
        for detection in sample.get("ground_truth", {}).get("detections", []):
            label = str(detection.get("label", "")).lower()
            # Voxel51's hard-hat dataset uses "head" for the visible head
            # class. It is the useful positive signal for no_helmet here;
            # absence of a vest is intentionally not converted to a label.
            mapped = {"person": 0, "helmet": 1, "head": 3}.get(label)
            if mapped is None:
                continue
            box = detection.get("bounding_box", [])
            if len(box) != 4:
                continue
            x, y, width, height = (float(value) for value in box)
            lines.append(f"{mapped} {x + width / 2:.6f} {y + height / 2:.6f} {width:.6f} {height:.6f}")
            class_counts[mapped] += 1
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[f"{split}_images"] += 1
    return {"counts": dict(counts), "class_counts": {str(k): v for k, v in sorted(class_counts.items())}, "missing": missing}


def _prepare_shwd(raw_root: Path, target: Path, seed: int) -> dict:
    archive = raw_root / "VOC2028.zip"
    extracted = raw_root / "VOC2028"
    if not extracted.exists() and archive.exists() and not zipfile.is_zipfile(archive):
        return {"status": "skipped", "reason": "VOC2028.zip is incomplete or not a ZIP archive", "archive": str(archive)}
    if not extracted.exists() and archive.exists():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(raw_root)
    candidates = [raw_root / "VOC2028", raw_root / "VOC2028" / "VOC2028"]
    root = next((item for item in candidates if (item / "Annotations").exists() and (item / "JPEGImages").exists()), None)
    if root is None:
        raise FileNotFoundError(f"cannot find extracted SHWD VOC folders under {raw_root}")
    xml_files = sorted((root / "Annotations").glob("*.xml"))
    random.Random(seed + 1).shuffle(xml_files)
    split_at = {"train": int(len(xml_files) * 0.8), "val": int(len(xml_files) * 0.9)}
    counts = Counter()
    class_counts = Counter()
    missing = []
    for index, xml_path in enumerate(xml_files):
        split = "train" if index < split_at["train"] else "val" if index < split_at["val"] else "test"
        image_name = None
        try:
            tree = ElementTree.parse(xml_path)
            image_name = tree.findtext("filename") or f"{xml_path.stem}.jpg"
            lines = []
            size = tree.find("size")
            width = float(size.findtext("width")) if size is not None else 0.0
            height = float(size.findtext("height")) if size is not None else 0.0
            if width <= 0 or height <= 0:
                continue
            for item in tree.findall("object"):
                name = (item.findtext("name") or "").strip().lower()
                mapped = {"hat": 1, "helmet": 1, "person": 3}.get(name)
                box = item.find("bndbox")
                if mapped is None or box is None:
                    continue
                xmin = float(box.findtext("xmin")); ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax")); ymax = float(box.findtext("ymax"))
                x = max(0.0, min(1.0, (xmin + xmax) / 2 / width))
                y = max(0.0, min(1.0, (ymin + ymax) / 2 / height))
                w = max(0.0, min(1.0, (xmax - xmin) / width))
                h = max(0.0, min(1.0, (ymax - ymin) / height))
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{mapped} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                class_counts[mapped] += 1
        except (ElementTree.ParseError, ValueError):
            continue
        image_path = root / "JPEGImages" / image_name
        if not image_path.exists():
            image_path = next(iter((root / "JPEGImages").glob(f"{xml_path.stem}.*")), None)
        if image_path is None or not image_path.exists():
            missing.append(str(xml_path))
            continue
        key = f"shwd/{image_path.name}"
        stem = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        target_image = target / "images" / split / f"shwd_{stem}{image_path.suffix.lower()}"
        target_label = target / "labels" / split / f"{target_image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[f"{split}_images"] += 1
    return {"counts": dict(counts), "class_counts": {str(k): v for k, v in sorted(class_counts.items())}, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Construction-PPE with Voxel51 hard-hat data.")
    parser.add_argument("--source", default="datasets/raw/voxel51_hardhat")
    parser.add_argument("--output", default="datasets/processed_augmented")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    settings = get_settings()
    source = (PROJECT_ROOT / args.source).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    original_counts = _copy_tree(settings.processed_dir, output)
    external = _prepare_voxel51(source, output, args.seed)
    shwd = None
    shwd_root = settings.raw_dir / "shwd"
    if (shwd_root / "VOC2028.zip").exists() or (shwd_root / "VOC2028").exists():
        shwd = _prepare_shwd(shwd_root, output, args.seed)
    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for name, index in PROJECT_CLASSES.items()},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {
        "sources": {"construction_ppe": original_counts, "voxel51_hard_hat_detection": external},
        "class_mapping": {"person": "person", "helmet": "helmet", "head": "no_helmet", "vest": "vest", "shwd_hat": "helmet", "shwd_person_head": "no_helmet"},
        "license_note": "Voxel51 dataset card states CC0-1.0; SHWD repository states MIT but verify original dataset terms before redistribution.",
        "source_urls": ["https://huggingface.co/datasets/Voxel51/hard-hat-detection", "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset"],
    }
    if shwd is not None:
        summary["sources"]["shwd"] = shwd
    report_dir = settings.report_dir.parent / "augmented"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(settings, ["construction_ppe:datasets/raw/construction_ppe", "voxel51_hard_hat_detection:datasets/raw/voxel51_hardhat", "shwd:datasets/raw/shwd/VOC2028.zip (incomplete; skipped)"])
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

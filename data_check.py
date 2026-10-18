"""Validate SHWD image/label pairing, split isolation, and YOLO label ranges."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def inspect_split(root: Path, split: str, class_count: int) -> tuple[dict, list[str], set[str]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    errors: list[str] = []
    if not image_dir.is_dir() or not label_dir.is_dir():
        return {}, [f"{split}: missing {image_dir} or {label_dir}"], set()

    images = image_stems(image_dir)
    labels = {path.stem for path in label_dir.glob("*.txt")}
    missing_labels = sorted(images - labels)
    missing_images = sorted(labels - images)
    if missing_labels:
        errors.append(f"{split}: {len(missing_labels)} images have no label; examples={missing_labels[:5]}")
    if missing_images:
        errors.append(f"{split}: {len(missing_images)} labels have no image; examples={missing_images[:5]}")

    class_instances: Counter[int] = Counter()
    invalid_rows = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except (ValueError, IndexError):
                invalid_rows += 1
                if len(errors) < 20:
                    errors.append(f"{label_path}:{line_number}: non-numeric or empty label row")
                continue
            if len(coords) != 4 or not 0 <= class_id < class_count or any(not 0.0 <= value <= 1.0 for value in coords):
                invalid_rows += 1
                if len(errors) < 20:
                    errors.append(f"{label_path}:{line_number}: invalid YOLO row {raw!r}")
                continue
            if coords[2] <= 0 or coords[3] <= 0:
                invalid_rows += 1
                if len(errors) < 20:
                    errors.append(f"{label_path}:{line_number}: box width/height must be positive")
                continue
            class_instances[class_id] += 1

    summary = {
        "images": len(images),
        "labels": len(labels),
        "paired": len(images & labels),
        "instances_by_class": dict(sorted(class_instances.items())),
        "invalid_rows": invalid_rows,
    }
    return summary, errors, images


def inspect_dataset(root: Path, class_count: int = 2) -> tuple[dict, list[str]]:
    report: dict = {"root": str(root.resolve()), "splits": {}}
    errors: list[str] = []
    split_stems: dict[str, set[str]] = {}
    for split in ("train", "val"):
        summary, split_errors, stems = inspect_split(root, split, class_count)
        report["splits"][split] = summary
        errors.extend(split_errors)
        split_stems[split] = stems

    overlap = sorted(split_stems["train"] & split_stems["val"])
    report["train_val_overlap"] = len(overlap)
    if overlap:
        errors.append(f"train/val share {len(overlap)} image stems; examples={overlap[:5]}")

    voc_root = root / "VOC2028"
    if voc_root.is_dir():
        report["voc_source"] = {
            "images": len(image_stems(voc_root / "JPEGImages")),
            "annotations": len(list((voc_root / "Annotations").glob("*.xml"))),
        }
    report["error_count"] = len(errors)
    return report, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--json", type=Path, help="Optional JSON report output")
    parser.add_argument("--strict", action="store_true", help="Return a failing exit code when issues are found")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, errors = inspect_dataset(args.root, args.classes)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    for error in errors:
        print(f"ERROR: {error}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

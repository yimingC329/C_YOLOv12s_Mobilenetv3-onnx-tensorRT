"""Run the same detection pipeline with a PyTorch, ONNX, or TensorRT model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mobile_yolo.runtime import load_yolo


SUPPORTED_SUFFIXES = {".pt", ".onnx", ".engine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Model file: .pt, .onnx, or .engine")
    parser.add_argument("source", help="Image, directory, video, stream URL, or camera index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/deploy"))
    parser.add_argument("--name")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def normalize_source(source: str):
    """Convert a plain integer to a camera index without changing file paths."""
    return int(source) if source.isdecimal() and not Path(source).exists() else source


def main() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(f"Model artifact not found: {args.artifact}")
    if args.artifact.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Supported model suffixes: {sorted(SUPPORTED_SUFFIXES)}")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf and --iou must be between 0 and 1")

    model = load_yolo(args.artifact, task="detect")
    run_name = args.name or args.artifact.stem
    results = model.predict(
        source=normalize_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        save=not args.no_save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=str(args.project),
        name=run_name,
        verbose=False,
    )

    processed = 0
    total_detections = 0
    actual_save_dir: Path | None = None
    for result in results:
        processed += 1
        if getattr(result, "save_dir", None):
            actual_save_dir = Path(result.save_dir)
        classes = result.boxes.cls.int().tolist() if result.boxes is not None else []
        total_detections += len(classes)
        counts: dict[str, int] = {}
        for class_id in classes:
            if isinstance(result.names, dict):
                class_name = str(result.names.get(class_id, class_id))
            elif 0 <= class_id < len(result.names):
                class_name = str(result.names[class_id])
            else:
                class_name = str(class_id)
            counts[class_name] = counts.get(class_name, 0) + 1
        summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "no detections"
        print(f"{result.path}: {summary}")

    print(f"Processed {processed} frame(s); detections={total_detections}")
    if not args.no_save:
        save_dir = actual_save_dir or (args.project / run_name)
        print(f"Saved under: {save_dir.resolve()}")


if __name__ == "__main__":
    main()

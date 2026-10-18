"""Validate PyTorch, ONNX, and TensorRT artifacts on the same dataset split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mobile_yolo.runtime import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("runs/deployment/validation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    missing = [str(path) for path in args.artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))

    rows: list[dict[str, object]] = []
    for artifact in args.artifacts:
        print(f"Validating {artifact} ...")
        model = load_yolo(artifact, task="detect")
        metrics = model.val(
            data=str(args.data.resolve()),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )
        speed = getattr(metrics, "speed", {}) or {}
        row = {
            "artifact": str(artifact.resolve()),
            "backend": artifact.suffix.lower().lstrip("."),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
        }
        rows.append(row)
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved validation comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()

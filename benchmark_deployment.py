"""Benchmark end-to-end batch-1 latency for PT, ONNX, and TensorRT artifacts."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from mobile_yolo.runtime import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("runs/deployment/latency.csv"))
    return parser.parse_args()


def synchronize(device: str) -> None:
    import torch

    if device.lower() != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid image size, warmup count, or iteration count")
    missing = [str(path) for path in args.artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))

    sample = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    for artifact in args.artifacts:
        print(f"Benchmarking {artifact} ...")
        model = load_yolo(artifact, task="detect")
        predict_args = {
            "source": sample,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "save": False,
            "verbose": False,
        }
        for _ in range(args.warmup):
            model.predict(**predict_args)
        synchronize(args.device)

        inference_times: list[float] = []
        start = time.perf_counter()
        for _ in range(args.iterations):
            result = model.predict(**predict_args)[0]
            inference_times.append(float(result.speed.get("inference", 0.0)))
        synchronize(args.device)
        elapsed = time.perf_counter() - start

        wall_ms = elapsed * 1000.0 / args.iterations
        row = {
            "artifact": str(artifact.resolve()),
            "backend": artifact.suffix.lower().lstrip("."),
            "imgsz": args.imgsz,
            "iterations": args.iterations,
            "wall_latency_ms": wall_ms,
            "wall_fps": 1000.0 / wall_ms,
            "backend_inference_ms": sum(inference_times) / len(inference_times),
            "size_mb": artifact.stat().st_size / (1024**2),
        }
        rows.append(row)
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved deployment benchmark: {args.output.resolve()}")


if __name__ == "__main__":
    main()

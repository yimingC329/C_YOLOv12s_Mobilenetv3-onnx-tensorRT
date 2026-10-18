"""Evaluate any baseline or MobileNetV3 checkpoint on the fixed validation split."""

from __future__ import annotations

import argparse

from mobile_yolo.runtime import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", default="runs/ablation/e1_yolo12s/weights/best.pt")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_yolo(args.weights)
    val_args = dict(data=args.data, split="val", imgsz=args.imgsz, batch=args.batch)
    if args.device is not None:
        val_args["device"] = args.device
    metrics = model.val(**val_args)

    print("=" * 56)
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print("=" * 56)
    names = model.names.values() if isinstance(model.names, dict) else model.names
    for index, name in enumerate(names):
        print(f"{name}: AP50={metrics.box.ap50[index]:.4f}, AP={metrics.box.ap[index]:.4f}")


if __name__ == "__main__":
    main()

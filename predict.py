"""Run helmet detection with either a baseline or MobileNetV3 checkpoint."""

from __future__ import annotations

import argparse

from mobile_yolo.runtime import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, directory, video, stream, or camera index")
    parser.add_argument("--weights", default="runs/ablation/yolov12s_baseline/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_yolo(args.weights)
    predict_args = dict(source=args.source, conf=args.conf, iou=args.iou, save=True, save_txt=True)
    if args.device is not None:
        predict_args["device"] = args.device
    results = model.predict(**predict_args)

    for result in results:
        classes = result.boxes.cls.tolist() if result.boxes is not None else []
        helmets = sum(int(class_id) == 0 for class_id in classes)
        uncovered_heads = sum(int(class_id) == 1 for class_id in classes)
        print(f"检测到: {helmets} 人佩戴安全帽, {uncovered_heads} 人未佩戴安全帽")
        if uncovered_heads:
            print("警告：存在未佩戴安全帽人员！")


if __name__ == "__main__":
    main()

"""Train one MobileNetV3 structural-ablation model under fixed settings."""

from __future__ import annotations

import argparse

from mobile_yolo.experiments import build_experiment_model, get_experiment


VARIANT_TO_EXPERIMENT = {"a": "E2", "b": "E3", "c": "E4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANT_TO_EXPERIMENT, default="b")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--fraction", type=float, help="Use a fraction of the dataset for a smoke run")
    parser.add_argument("--device", help="Ultralytics device value, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--name")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random initialization instead of transferring compatible YOLOv12s weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_id = VARIANT_TO_EXPERIMENT[args.variant]
    train_args, experiment = get_experiment(experiment_id)
    for key in ("epochs", "batch", "imgsz", "fraction", "device", "workers"):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value
    if args.name:
        experiment["name"] = args.name

    model = build_experiment_model(experiment, use_pretrained=not args.no_pretrained)
    results = model.train(**train_args, name=experiment["name"])
    save_dir = getattr(results, "save_dir", "runs/ablation")
    print(f"{experiment['label']} training complete: {save_dir}")


if __name__ == "__main__":
    main()

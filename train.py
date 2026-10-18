"""Train the unchanged YOLOv12s baseline (experiment E1)."""

from __future__ import annotations

import argparse

from mobile_yolo.experiments import build_experiment_model, get_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="Override the fair-comparison epoch count")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override input size")
    parser.add_argument("--fraction", type=float, help="Use a fraction of the dataset for a smoke run")
    parser.add_argument("--device", help="Ultralytics device value, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, help="Override dataloader worker count")
    parser.add_argument("--name", help="Override the run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_args, experiment = get_experiment("E1")
    for key in ("epochs", "batch", "imgsz", "fraction", "device", "workers", "name"):
        value = getattr(args, key)
        if value is not None:
            if key == "name":
                experiment["name"] = value
            else:
                train_args[key] = value

    model = build_experiment_model(experiment)
    results = model.train(**train_args, name=experiment["name"])
    save_dir = getattr(results, "save_dir", "runs/ablation")
    print(f"Baseline training complete: {save_dir}")


if __name__ == "__main__":
    main()

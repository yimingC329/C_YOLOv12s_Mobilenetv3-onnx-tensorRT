"""Train and/or summarize the E0-E4 fair-comparison experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark import benchmark_model
from mobile_yolo.experiments import (
    build_experiment_model,
    get_experiment,
    load_experiment_config,
    resolve_project_path,
)
from mobile_yolo.runtime import load_yolo


DEFAULT_IDS = ("E0", "E1", "E2", "E3", "E4")


def train_experiments(ids: list[str], device: str | None) -> dict[str, Path]:
    trained: dict[str, Path] = {}
    for experiment_id in ids:
        train_args, experiment = get_experiment(experiment_id)
        if device is not None:
            train_args["device"] = device
        print(f"\n{'=' * 72}\n{experiment_id}: {experiment['label']}\n{'=' * 72}")
        model = build_experiment_model(experiment)
        results = model.train(**train_args, name=experiment["name"])
        save_dir = Path(results.save_dir)
        trained[experiment_id] = save_dir / "weights" / "best.pt"
    return trained


def default_checkpoint(experiment: dict, project: str) -> Path:
    return resolve_project_path(project) / experiment["name"] / "weights" / "best.pt"


def evaluate_checkpoint(
    experiment_id: str,
    checkpoint: Path,
    data: str,
    imgsz: int,
    batch: int,
    device: str | None,
    warmup: int,
    iterations: int,
) -> dict:
    model = load_yolo(checkpoint)
    val_args = dict(data=data, split="val", imgsz=imgsz, batch=batch, plots=True)
    if device is not None:
        val_args["device"] = device
    metrics = model.val(**val_args)
    speed = benchmark_model(
        checkpoint,
        imgsz=imgsz,
        batch=1,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    return {
        "experiment": experiment_id,
        "checkpoint": str(checkpoint),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "params_m": speed["params_m"],
        "gflops": speed["gflops"],
        "fps": speed["fps"],
        "latency_ms": speed["latency_ms"],
        "model_size_mb": speed["model_size_mb"],
        "peak_gpu_memory_mb": speed["peak_gpu_memory_mb"],
        "status": "ok",
    }


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "label",
        "checkpoint",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "params_m",
        "gflops",
        "fps",
        "latency_ms",
        "model_size_mb",
        "peak_gpu_memory_mb",
        "status",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], output: Path) -> None:
    complete = [row for row in rows if row.get("status") == "ok"]
    if not complete:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Warning: matplotlib is unavailable; comparison plot was skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [row["label"] for row in complete]
    axes[0].bar(labels, [row["map50"] for row in complete], color="#2F5597")
    axes[0].set_title("Validation mAP@0.5")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].scatter(
        [row["latency_ms"] for row in complete],
        [row["map50"] for row in complete],
        s=[max(40, row["params_m"] * 12) for row in complete],
        color="#C55A11",
    )
    for row in complete:
        axes[1].annotate(row["experiment"], (row["latency_ms"], row["map50"]), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Raw forward latency (ms/image)")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("Accuracy-efficiency trade-off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train selected experiments before summarizing")
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_IDS), choices=DEFAULT_IDS)
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison/model_comparison.csv"))
    parser.add_argument("--plot", type=Path, default=Path("runs/comparison/model_comparison.png"))
    parser.add_argument("--strict", action="store_true", help="Fail when any checkpoint is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    ids = [value.upper() for value in args.experiments]
    trained = train_experiments(ids, args.device) if args.train else {}
    common = config["common_train"]
    rows: list[dict] = []

    for experiment_id in ids:
        experiment = config["experiments"][experiment_id]
        checkpoint = trained.get(experiment_id) or default_checkpoint(experiment, common["project"])
        if not checkpoint.is_file():
            row = {
                "experiment": experiment_id,
                "label": experiment["label"],
                "checkpoint": str(checkpoint),
                "status": "missing checkpoint",
            }
            rows.append(row)
            print(f"Skipping {experiment_id}: {checkpoint} not found")
            continue
        row = evaluate_checkpoint(
            experiment_id,
            checkpoint,
            data=common["data"],
            imgsz=int(common["imgsz"]),
            batch=int(common["batch"]),
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        row["label"] = experiment["label"]
        rows.append(row)

    write_csv(rows, args.output)
    write_plot(rows, args.plot)
    print(f"Comparison table: {args.output}")
    if args.strict and any(row.get("status") != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

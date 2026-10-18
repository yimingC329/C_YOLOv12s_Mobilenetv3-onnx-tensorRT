"""Benchmark parameters, FLOPs, latency, FPS, memory, and checkpoint size."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mobile_yolo.runtime import load_yolo


def normalize_device(torch, requested: str | None):
    if requested in (None, ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        requested = f"cuda:{requested}"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def benchmark_model(
    source: str | Path,
    imgsz: int = 640,
    batch: int = 1,
    device: str | None = None,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    """Benchmark raw PyTorch model forward passes under controlled conditions."""
    import torch

    if imgsz <= 0 or batch <= 0 or warmup < 0 or iterations <= 0:
        raise ValueError("imgsz, batch, and iterations must be positive; warmup cannot be negative")

    yolo = load_yolo(source)
    run_device = normalize_device(torch, device)
    core = yolo.model.to(run_device).eval()
    sample = torch.zeros(batch, 3, imgsz, imgsz, device=run_device)
    params = sum(parameter.numel() for parameter in core.parameters())

    gflops = None
    try:
        from thop import profile

        multiply_adds, _ = profile(core, inputs=(sample,), verbose=False)
        gflops = 2.0 * multiply_adds / 1e9 / batch
    except Exception as exc:  # FLOP support varies across Ultralytics releases
        print(f"Warning: FLOPs unavailable: {exc}")

    if run_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(run_device)

    with torch.inference_mode():
        for _ in range(warmup):
            core(sample)
        if run_device.type == "cuda":
            torch.cuda.synchronize(run_device)

        start = time.perf_counter()
        for _ in range(iterations):
            core(sample)
        if run_device.type == "cuda":
            torch.cuda.synchronize(run_device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / (iterations * batch)
    fps = 1000.0 / latency_ms
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(run_device) / (1024**2) if run_device.type == "cuda" else None
    )
    source_path = Path(source)
    model_size_mb = (
        source_path.stat().st_size / (1024**2)
        if source_path.is_file() and source_path.suffix.lower() == ".pt"
        else None
    )

    return {
        "source": str(source),
        "device": str(run_device),
        "imgsz": imgsz,
        "batch": batch,
        "params": params,
        "params_m": params / 1e6,
        "gflops": gflops,
        "latency_ms": latency_ms,
        "fps": fps,
        "peak_gpu_memory_mb": peak_memory_mb,
        "model_size_mb": model_size_mb,
        "latency_scope": "raw_model_forward",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Checkpoint (.pt) or model YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_model(
        args.source,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

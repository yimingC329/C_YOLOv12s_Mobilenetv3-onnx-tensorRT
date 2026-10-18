"""Export a trained YOLO checkpoint to a TensorRT engine and smoke-test it."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from mobile_yolo.runtime import export_precision_args, load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to a trained best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0", help="CUDA device index, normally 0")
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace in GiB")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument("--nms", action="store_true", help="Embed NMS in the engine")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp16", dest="precision", action="store_const", const="fp16")
    precision.add_argument("--fp32", dest="precision", action="store_const", const="fp32")
    precision.add_argument("--int8", dest="precision", action="store_const", const="int8")
    parser.set_defaults(precision="fp16")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/data.yaml"),
        help="Dataset YAML used for INT8 calibration",
    )
    parser.add_argument("--output", type=Path, help="Optional destination .engine path")
    parser.add_argument("--skip-verify", action="store_true", help="Skip one-image engine smoke test")
    return parser.parse_args()


def require_tensorrt(precision: str) -> str:
    try:
        import tensorrt as trt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT is not installed. Activate .venv and run "
            "`python -m pip install -r requirements-deploy.txt`."
        ) from exc
    major = int(trt.__version__.split(".", 1)[0])
    if major >= 11 and precision in {"fp16", "int8"}:
        try:
            import modelopt  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorRT 11 requires NVIDIA ModelOpt for FP16/INT8 export. "
                "Run `python -m pip install \"nvidia-modelopt[onnx]>=0.44\"`."
            ) from exc
    return trt.__version__


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if args.imgsz <= 0 or args.batch <= 0 or args.workspace <= 0:
        raise ValueError("--imgsz, --batch, and --workspace must be positive")
    if args.precision == "int8" and not args.data.is_file():
        raise FileNotFoundError(f"INT8 calibration dataset YAML not found: {args.data}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a CUDA-enabled PyTorch environment")
    trt_version = require_tensorrt(args.precision)
    device_index = int(str(args.device).split(":")[-1])
    print(f"TensorRT {trt_version}; CUDA device: {torch.cuda.get_device_name(device_index)}")

    model = load_yolo(args.weights, task="detect")
    export_args: dict[str, object] = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workspace": args.workspace,
        "dynamic": args.dynamic,
        "nms": args.nms,
    }
    export_args.update(export_precision_args(args.precision))
    if args.precision == "int8":
        export_args["data"] = str(args.data.resolve())

    exported = Path(model.export(**export_args))
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics reported an export path that does not exist: {exported}")

    if args.output:
        destination = args.output.resolve()
        if destination.suffix.lower() != ".engine":
            raise ValueError("--output must end with .engine")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != destination:
            shutil.copy2(exported, destination)
        exported = destination

    if not args.skip_verify:
        engine = load_yolo(exported, task="detect")
        sample = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        engine.predict(sample, imgsz=args.imgsz, device=args.device, verbose=False)
        print("TensorRT one-image smoke test passed")

    print(f"Exported: {exported.resolve()}")


if __name__ == "__main__":
    main()

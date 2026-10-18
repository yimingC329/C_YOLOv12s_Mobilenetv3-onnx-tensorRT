"""Export a trained YOLOv12 or MobileNetV3 checkpoint to ONNX and verify it."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mobile_yolo.runtime import export_precision_args, load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to a trained best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--nms", action="store_true", help="Embed NMS in the exported graph")
    parser.add_argument("--output", type=Path, help="Optional destination .onnx path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--imgsz and --batch must be positive")

    model = load_yolo(args.weights)
    export_args = dict(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=args.nms,
    )
    if args.half:
        export_args.update(export_precision_args("fp16"))
    if args.device is not None:
        export_args["device"] = args.device
    exported = Path(model.export(**export_args))
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics reported an export path that does not exist: {exported}")

    if args.output:
        destination = args.output.resolve()
        if destination.suffix.lower() != ".onnx":
            raise ValueError("--output must end with .onnx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != destination:
            shutil.copy2(exported, destination)
        exported = destination

    try:
        import onnx

        graph = onnx.load(str(exported))
        onnx.checker.check_model(graph)
        input_info = [f"{item.name}: {[dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]}" for item in graph.graph.input]
        output_info = [f"{item.name}: {[dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]}" for item in graph.graph.output]
        print("ONNX structural validation passed")
        print("Inputs:  " + "; ".join(input_info))
        print("Outputs: " + "; ".join(output_info))
    except ModuleNotFoundError:
        print("Warning: onnx is not installed; structural validation was skipped")
    print(f"Exported: {exported.resolve()}")


if __name__ == "__main__":
    main()

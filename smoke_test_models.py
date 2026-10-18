"""Build all local model YAMLs and verify P3/P4/P5 feature-map dimensions."""

from __future__ import annotations

from pathlib import Path

from mobile_yolo.runtime import load_yolo


MODEL_CONFIGS = (
    Path("models/yolov12s_helmet.yaml"),
    Path("models/yolov12s_mbv3_a.yaml"),
    Path("models/yolov12s_mbv3_b.yaml"),
    Path("models/yolov12s_mbv3_c.yaml"),
)


def main() -> None:
    import torch

    expected_spatial = {4: 32, 6: 16, 8: 8}
    for config in MODEL_CONFIGS:
        model = load_yolo(config)
        core = model.model.eval()
        features: dict[int, tuple[int, ...]] = {}
        hooks = [
            core.model[index].register_forward_hook(
                lambda _module, _inputs, output, layer=index: features.__setitem__(layer, tuple(output.shape))
            )
            for index in expected_spatial
        ]
        with torch.inference_mode():
            core(torch.zeros(1, 3, 256, 256))
        for hook in hooks:
            hook.remove()

        for layer, size in expected_spatial.items():
            shape = features[layer]
            if shape[-2:] != (size, size):
                raise AssertionError(f"{config}: layer {layer} shape {shape} != expected spatial size {size}")
        for layer in (6, 8):
            if core.model[layer].__class__.__name__ != "A2C2f":
                raise AssertionError(f"{config}: YOLOv12 attention missing at layer {layer}")
        params = sum(parameter.numel() for parameter in core.parameters())
        print(f"PASS {config}: params={params / 1e6:.3f}M, features={features}")


if __name__ == "__main__":
    main()

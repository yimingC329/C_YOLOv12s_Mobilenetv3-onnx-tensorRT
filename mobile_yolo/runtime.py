"""Shared, dependency-friendly Ultralytics loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def export_precision_args(precision: str) -> dict[str, object]:
    """Return precision kwargs compatible with old and new Ultralytics APIs.

    Ultralytics versions up to the 8.3 line used ``half`` and ``int8``.
    Newer releases use one ``quantize`` argument (16, 8, or None).  Keeping
    this translation here lets the project retain its broad <9 requirement.
    """
    normalized = precision.strip().lower()
    if normalized not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Unsupported export precision: {precision}")

    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {} if normalized == "fp32" else {"quantize": 16 if normalized == "fp16" else 8}
    if normalized == "fp16":
        return {"half": True}
    if normalized == "int8":
        return {"int8": True}
    return {}


def require_yolo():
    """Return ``ultralytics.YOLO`` with custom modules registered."""
    # Keep framework settings inside the project. This avoids writes to a
    # locked home/AppData directory on managed workstations and servers.
    config_dir = ROOT / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    matplotlib_dir = ROOT / ".cache" / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Activate the project environment and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    from .registry import register_ultralytics_modules

    register_ultralytics_modules()
    return YOLO


def load_yolo(
    source: str | Path,
    pretrained: str | Path | None = None,
    task: str | None = None,
):
    """Build/load a YOLO model or deployment artifact.

    ``task`` is useful for exported ONNX/TensorRT files whose metadata is not
    available yet.  It is optional for normal PyTorch checkpoints.
    """
    YOLO = require_yolo()
    source = str(source)
    model = YOLO(source, task=task)
    if pretrained:
        model.load(str(pretrained))
    return model

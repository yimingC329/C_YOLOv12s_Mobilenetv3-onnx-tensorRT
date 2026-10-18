"""Register project-local model layers with an installed Ultralytics package."""

from __future__ import annotations


def register_ultralytics_modules() -> None:
    """Expose custom layers to the namespace used by Ultralytics ``parse_model``.

    The operation is idempotent and modifies only the current Python process;
    it never writes into site-packages.
    """
    import ultralytics.nn.modules as ultralytics_modules
    import ultralytics.nn.tasks as ultralytics_tasks

    from .mobilenetv3 import InvertedResidual, MobileNetV3Stage, SqueezeExcitation

    custom_modules = (InvertedResidual, MobileNetV3Stage, SqueezeExcitation)
    for module in custom_modules:
        setattr(ultralytics_tasks, module.__name__, module)
        setattr(ultralytics_modules, module.__name__, module)

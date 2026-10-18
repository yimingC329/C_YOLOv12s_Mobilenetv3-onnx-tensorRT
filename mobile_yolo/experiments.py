"""Experiment configuration loader shared by training and reporting scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


def load_experiment_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt first.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or "experiments" not in config:
        raise ValueError(f"Invalid experiment config: {config_path}")
    return config


def get_experiment(experiment_id: str, path: str | Path = DEFAULT_CONFIG) -> tuple[dict, dict]:
    config = load_experiment_config(path)
    common = deepcopy(config["common_train"])
    common["data"] = str(resolve_project_path(common["data"]))
    common["project"] = str(resolve_project_path(common["project"]))
    key = experiment_id.upper()
    try:
        experiment = deepcopy(config["experiments"][key])
    except KeyError as exc:
        choices = ", ".join(config["experiments"])
        raise KeyError(f"Unknown experiment {experiment_id!r}; choose one of: {choices}") from exc
    return common, experiment


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_experiment_model(experiment: dict, use_pretrained: bool = True):
    from .runtime import load_yolo

    model_source = experiment["model"]
    if str(model_source).lower().endswith((".yaml", ".yml")):
        model_source = resolve_project_path(model_source)
    pretrained = experiment.get("pretrained") if use_pretrained else None
    return load_yolo(model_source, pretrained=pretrained)

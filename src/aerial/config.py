"""Config loading.

A run is described by `configs/base.yaml` plus a small run config that
overrides only what differs. Keeping the shared settings in one place means
three runs differ by exactly one value - the epoch count - rather than by
whatever drifted between three near-identical files.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. Override wins; nested dicts merge rather than replace."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_override(item: str) -> tuple[list[str], Any]:
    """Turn `train.epochs=250` into (["train", "epochs"], 250)."""
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got: {item}")
    key, raw = item.split("=", 1)
    return key.split("."), yaml.safe_load(raw)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    config = copy.deepcopy(config)
    for item in overrides:
        path, value = parse_override(item)
        node = config
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return config


def load(run_config: Path, overrides: list[str] | None = None) -> dict:
    run_config = Path(run_config)
    if not run_config.exists():
        raise SystemExit(f"Config not found: {run_config}")

    base = yaml.safe_load(BASE_CONFIG.read_text()) if BASE_CONFIG.exists() else {}
    config = deep_merge(base, yaml.safe_load(run_config.read_text()) or {})

    if overrides:
        config = apply_overrides(config, overrides)

    for key in ("run_name", "model", "classes"):
        if key not in config:
            raise SystemExit(f"Config is missing required key: {key}")
    if "epochs" not in config.get("train", {}):
        raise SystemExit("Config is missing train.epochs")

    # Evaluating at a different resolution than training measures something
    # else. Catch it here rather than in the numbers.
    if config["val"]["imgsz"] != config["train"]["imgsz"]:
        raise SystemExit(
            f"val.imgsz ({config['val']['imgsz']}) must equal "
            f"train.imgsz ({config['train']['imgsz']})"
        )

    return config


def run_dir(config: dict) -> Path:
    return Path(config["output"]["root"]) / config["run_name"]

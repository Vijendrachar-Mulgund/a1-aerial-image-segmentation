#!/usr/bin/env python3
"""Train a run.

    uv run aerial-train --config configs/v0.1-100.yaml
    uv run aerial-train --config configs/v0.1-300.yaml --set train.batch=2

Outputs land in {output.root}/{run_name} - by default a folder inside your
Google Drive `aerial` folder, so a dropped Colab session costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from aerial import dataset
from aerial.config import load, run_dir


def set_determinism(seed: int) -> None:
    """Seed before the model is constructed, not after."""
    import os

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    if int(torch.__version__.split(".")[0]) < 2:
        raise SystemExit(
            f"torch {torch.__version__} cannot train deterministically. "
            "Ultralytics warns and continues non-deterministically below 2.0."
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment() -> dict:
    import platform

    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env |= {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        pass
    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return env


def read_metrics(metrics, class_names: list[str]) -> dict:
    """Semantic segmentation reports IoU, not mAP.

    Attribute names carry a leading underscore in current versions, so probe
    for public properties first and degrade to an empty dict rather than
    losing a completed run to a logging detail.
    """
    per_class = getattr(metrics, "per_class_iou", None)
    if per_class is None:
        per_class = getattr(metrics, "_per_class_iou", None)
    if per_class is None:
        print("  WARNING: could not read per-class IoU", file=sys.stderr)
        return {}

    pixel_acc = getattr(metrics, "_per_class_pixel_acc", None)
    pixels = getattr(metrics, "nt_per_class", None)
    images = getattr(metrics, "nt_per_image", None)

    labels = [*class_names, "background"]
    out = {}
    for i in range(len(per_class)):
        name = labels[i] if i < len(labels) else f"class_{i}"
        entry = {"iou": float(per_class[i])}
        if pixel_acc is not None and i < len(pixel_acc):
            entry["pixel_acc"] = float(pixel_acc[i])
        if pixels is not None and i < len(pixels):
            entry["pixels"] = int(pixels[i])
        if images is not None and i < len(images):
            entry["images"] = int(images[i])
        out[name] = entry

    # Background is the complement of everything else and sits very high.
    # Reporting it inside the mean would flatter every run.
    class_only = [out[n]["iou"] for n in class_names if n in out]
    out["_summary"] = {
        "miou_classes_only": (sum(class_only) / len(class_only) if class_only else None),
        "miou_including_background": float(getattr(metrics, "_miou", 0.0) or 0.0),
        "pixel_accuracy": float(getattr(metrics, "_pixel_accuracy", 0.0) or 0.0),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.batch=2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare and validate the dataset, then stop",
    )
    args = parser.parse_args()

    config = load(args.config, args.overrides)
    target = run_dir(config)
    print(f"Run: {config['run_name']}  ->  {target}")

    print("Preparing dataset...")
    data_yaml = dataset.prepare(config)
    if args.dry_run:
        print("Dry run complete.")
        return 0

    set_determinism(config["seed"])

    from ultralytics import YOLO

    model = YOLO(config["model"])

    train_args = dict(config["train"])
    train_args |= {
        "data": str(data_yaml),
        "seed": config["seed"],
        "project": config["output"]["root"],
        "name": config["run_name"],
        "exist_ok": False,
    }

    print(
        f"Training {config['model']} for {train_args['epochs']} epochs "
        f"at imgsz={train_args['imgsz']}"
    )
    model.train(**train_args)

    weights = target / "weights" / "best.pt"
    print(f"\nBenchmarking {weights}")
    best = YOLO(str(weights))
    metrics = best.val(split="test", **config["val"])
    results = read_metrics(metrics, config["classes"])

    manifest = {
        "run_name": config["run_name"],
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "dataset_yaml": str(data_yaml),
        "dataset_classes": yaml.safe_load(data_yaml.read_text()).get("names"),
        "metrics": results,
        "environment": environment(),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n{'class':14s} {'IoU':>7s} {'pixAcc':>8s} {'pixels':>14s} {'imgs':>5s}")
    for name, entry in results.items():
        if name.startswith("_"):
            continue
        print(
            f"{name:14s} {entry['iou']:7.3f} {entry.get('pixel_acc', float('nan')):8.3f} "
            f"{entry.get('pixels', 0):14,d} {entry.get('images', 0):5d}"
        )
    summary = results.get("_summary", {})
    if summary.get("miou_classes_only") is not None:
        print(f"\nmIoU (classes only): {summary['miou_classes_only']:.3f}")
    print(f"Manifest: {target / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark several trained models on one shared test split and plot the
comparison.

Designed to be pasted into a Colab notebook cell by cell, but runs as a
script too. Every model is evaluated on the same images at the same imgsz,
which is the only way the numbers mean anything next to each other.

Semantic segmentation reports IoU, not mAP. Higher is better for every
accuracy metric here; lower is better for latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG - edit these
# ---------------------------------------------------------------------------

DRIVE = "/content/drive/MyDrive/aerial"

MODELS = {
    "v0.1-100s  (90 img, s, 100ep)": f"{DRIVE}/v0.1-100-s/weights/best.pt",
    "v0.2-100s  (600 img, s, 100ep)": f"{DRIVE}/v0.2-100-s/weights/best.pt",
    "v0.2-300s  (600 img, s, 300ep)": f"{DRIVE}/v0.2-300-s/weights/best.pt",
    "v0.2-500s  (600 img, s, 500ep)": f"{DRIVE}/v0.2-500-s/weights/best.pt",
    "v0.2-300m (600 img, m, 300ep)": f"{DRIVE}/v0.2-300-m/weights/best.pt",
}

DATA_YAML = "/content/A1-Ariel-Image-Segmentation-3/data.yaml"
IMGSZ = 1280  # must match training for every model
BATCH = 2
DEVICE = 0
CLASSES = ["divider", "road", "sidewalk"]  # background is appended by the loader
OUT_DIR = Path(f"{DRIVE}/benchmark")

# Distinct colours and markers so the legend works in greyscale print too.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
MARKERS = ["o", "s", "^", "D", "v"]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract(metrics, classes: list[str]) -> dict:
    """Pull SemanticMetrics into a plain dict.

    Attribute names carry a leading underscore in current Ultralytics
    versions, so probe for public properties first and degrade gracefully
    rather than losing a completed evaluation to a naming change.
    """

    def get(*names, default=None):
        for n in names:
            v = getattr(metrics, n, None)
            if v is not None:
                return v
        return default

    iou = get("per_class_iou", "_per_class_iou", default=[])
    pacc = get("per_class_pixel_acc", "_per_class_pixel_acc", default=[])
    pixels = get("nt_per_class", default=[])
    images = get("nt_per_image", default=[])
    speed = get("speed", default={}) or {}

    labels = [*classes, "background"]
    per_class = {}
    for i in range(len(iou)):
        name = labels[i] if i < len(labels) else f"class_{i}"
        per_class[name] = {
            "iou": float(iou[i]),
            "pixel_acc": float(pacc[i]) if i < len(pacc) else float("nan"),
            "pixels": int(pixels[i]) if i < len(pixels) else 0,
            "images": int(images[i]) if i < len(images) else 0,
        }

    class_iou = [per_class[c]["iou"] for c in classes if c in per_class]
    return {
        "per_class": per_class,
        "miou_classes": float(np.mean(class_iou)) if class_iou else float("nan"),
        "miou_with_bg": float(get("miou", "_miou", default=float("nan"))),
        "pixel_accuracy": float(get("pixel_accuracy", "_pixel_accuracy", default=float("nan"))),
        "latency_ms": float(speed.get("inference", float("nan"))),
    }


def benchmark(models: dict, data_yaml: str, classes: list[str]) -> dict:
    from ultralytics import YOLO

    results = {}
    for name, path in models.items():
        if not Path(path).exists():
            print(f"SKIP {name}: weights not found at {path}")
            continue

        print(f"\nEvaluating {name}")
        model = YOLO(path)
        metrics = model.val(
            data=data_yaml,
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            verbose=False,
            plots=False,
        )
        entry = extract(metrics, classes)

        # Capacity, for the accuracy-per-compute view.
        try:
            info = model.model.info(detailed=False, verbose=False)
            entry["params"] = int(info[1])
            entry["gflops"] = float(info[3])
        except Exception:
            entry["params"] = entry["gflops"] = float("nan")

        results[name] = entry
        print(f"  mIoU (classes only): {entry['miou_classes']:.4f}")
    return results


def to_frame(results: dict, classes: list[str]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {
            "model": name,
            "mIoU_classes": r["miou_classes"],
            "mIoU_with_bg": r["miou_with_bg"],
            "pixel_accuracy": r["pixel_accuracy"],
            "latency_ms": r["latency_ms"],
            "params_M": r.get("params", float("nan")) / 1e6,
            "gflops": r.get("gflops", float("nan")),
        }
        for c in [*classes, "background"]:
            entry = r["per_class"].get(c, {})
            row[f"iou_{c}"] = entry.get("iou", float("nan"))
            row[f"pixacc_{c}"] = entry.get("pixel_acc", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _better(ax, text: str) -> None:
    """Annotate which direction is good. Easy to get backwards when reading
    a chart cold, and IoU vs latency point opposite ways."""
    ax.text(
        0.995,
        1.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#444",
    )


def plot_all(df: pd.DataFrame, classes: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = df["model"].tolist()
    colours = PALETTE[: len(names)]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.22)

    # 1. Per-class IoU, grouped
    ax = fig.add_subplot(gs[0, :])
    shown = [*classes, "background"]
    x = np.arange(len(shown))
    width = 0.8 / len(names)
    for i, (name, colour) in enumerate(zip(names, colours, strict=False)):
        vals = [df.loc[df.model == name, f"iou_{c}"].iloc[0] for c in shown]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=name,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(shown)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class IoU on the shared test split", fontweight="bold", pad=18)
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, ls="--", lw=0.8, color="#999")
    _better(ax, "higher is better")

    # 2. mIoU, with and without background
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(names))
    ax.bar(
        x - 0.2, df["mIoU_classes"], 0.4, label="classes only", color="#4C72B0", edgecolor="white"
    )
    ax.bar(
        x + 0.2,
        df["mIoU_with_bg"],
        0.4,
        label="incl. background",
        color="#BBBBBB",
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([n.split()[0] for n in names], rotation=20, ha="right")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1)
    ax.set_title("Mean IoU", fontweight="bold", pad=18)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _better(ax, "higher is better")

    # 3. Per-class pixel accuracy
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(shown))
    for i, (name, colour) in enumerate(zip(names, colours, strict=False)):
        vals = [df.loc[df.model == name, f"pixacc_{c}"].iloc[0] for c in shown]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(shown)
    ax.set_ylabel("pixel accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class pixel accuracy", fontweight="bold", pad=18)
    ax.grid(axis="y", alpha=0.3)
    _better(ax, "higher is better")

    # 4. Accuracy against compute
    ax = fig.add_subplot(gs[2, 0])
    for i, (name, colour, marker) in enumerate(zip(names, colours, MARKERS, strict=False)):  # noqa: B007
        row = df[df.model == name].iloc[0]
        ax.scatter(
            row["gflops"],
            row["mIoU_classes"],
            s=190,
            color=colour,
            marker=marker,
            label=name,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("mIoU (classes only)")
    ax.set_title("Accuracy per unit of compute", fontweight="bold", pad=18)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    _better(ax, "up and to the left is better")

    # 5. Latency
    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(len(names))
    ax.bar(x, df["latency_ms"], 0.55, color=colours, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split()[0] for n in names], rotation=20, ha="right")
    ax.set_ylabel("ms per image")
    ax.set_title(f"Inference latency at imgsz={IMGSZ}", fontweight="bold", pad=18)
    ax.grid(axis="y", alpha=0.3)
    _better(ax, "lower is better")

    fig.suptitle(
        "Model comparison - semantic segmentation of road features",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out_dir / "comparison.png", dpi=160, bbox_inches="tight")
    plt.show()


def plot_per_class_trend(df: pd.DataFrame, classes: list[str], out_dir: Path) -> None:
    """One line per class across models. Makes divergent behaviour obvious -
    a change that helps road while hurting divider shows up here and is
    invisible in the mean."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    for c, marker in zip(classes, MARKERS, strict=False):
        ax.plot(x, df[f"iou_{c}"], marker=marker, linewidth=2, markersize=9, label=c)
    ax.plot(
        x,
        df["mIoU_classes"],
        "k--",
        marker="*",
        markersize=13,
        linewidth=1.6,
        label="mIoU (classes)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class IoU across models", fontweight="bold", pad=18)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _better(ax, "higher is better")
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_trend.png", dpi=160, bbox_inches="tight")
    plt.show()


def main() -> None:
    results = benchmark(MODELS, DATA_YAML, CLASSES)
    if not results:
        raise SystemExit("No models evaluated - check the paths in MODELS")

    df = to_frame(results, CLASSES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "benchmark.csv", index=False)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))

    plot_all(df, CLASSES, OUT_DIR)
    plot_per_class_trend(df, CLASSES, OUT_DIR)

    cols = ["model", "mIoU_classes", *[f"iou_{c}" for c in CLASSES], "latency_ms"]
    print("\n" + df[cols].to_string(index=False, float_format="%.4f"))

    best = df.loc[df["mIoU_classes"].idxmax()]
    print(f"\nBest by mIoU (classes only): {best['model']}  {best['mIoU_classes']:.4f}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

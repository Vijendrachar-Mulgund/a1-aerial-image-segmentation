"""Dataset acquisition and preparation.

Two sources, one interface: `prepare(config)` returns the path to a usable
data.yaml regardless of whether the dataset came from Roboflow or already
exists on disk. Swapping datasets is a config change, not a code change.

Everything here is defensive. A Roboflow export needs three fixes before the
semantic loader will read it, and each one fails in a way that is hard to
diagnose from the training error alone.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def prepare(config: dict) -> Path:
    dataset = config["dataset"]
    source = dataset.get("source", "roboflow")

    if source == "roboflow":
        root = _download_roboflow(dataset)
    elif source == "local":
        if not dataset.get("path"):
            raise SystemExit("dataset.source is 'local' but dataset.path is not set")
        root = Path(dataset["path"]).resolve()
        if not root.exists():
            raise SystemExit(f"dataset.path does not exist: {root}")
    else:
        raise SystemExit(f"Unknown dataset.source: {source}")

    data_yaml = _repair_yaml(root)
    _validate(root, data_yaml, config)
    return data_yaml


def _download_roboflow(dataset: dict) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("roboflow is not installed. Run: uv sync --group data") from exc

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Export it, or use Colab Secrets:\n"
            "    from google.colab import userdata\n"
            "    os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')"
        )

    download_dir = Path(dataset.get("download_dir", "/content/datasets"))
    download_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(dataset["workspace"]).project(dataset["project"])
    handle = project.version(int(dataset["version"]))
    downloaded = handle.download(
        dataset.get("format", "yolov8"), location=str(download_dir / dataset["project"])
    )
    return Path(downloaded.location).resolve()


def _repair_yaml(root: Path) -> Path:
    """Make a Roboflow export readable by the semantic loader.

    Three fixes, each guarding a real failure mode:

    1. A `masks/` folder at the dataset root switches the loader to PNG-mask
       mode even when masks_dir is absent from the YAML, and it then looks
       for masks that are not there. Rename it out of the way.
    2. Roboflow writes relative paths that resolve against the caller's
       working directory. Make them absolute.
    3. Roboflow writes `names` as a list; the semantic loader wants the
       {index: name} mapping.
    """
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml found at {root}")

    stray_masks = root / "masks"
    if stray_masks.is_dir():
        stray_masks.rename(root / "_masks_unused")
        print(f"  renamed stray masks/ folder at {root}")

    config = yaml.safe_load(data_yaml.read_text())
    config["path"] = str(root)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / folder / "images").is_dir():
            config[split] = f"{folder}/images"
        else:
            config.pop(split, None)

    if isinstance(config.get("names"), list):
        config["names"] = dict(enumerate(config["names"]))

    # Presence of this key alone selects PNG-mask mode.
    config.pop("masks_dir", None)

    yaml.safe_dump(config, data_yaml.open("w"), sort_keys=False)
    return data_yaml


def _validate(root: Path, data_yaml: Path, config: dict) -> None:
    data = yaml.safe_load(data_yaml.read_text())

    names = data.get("names", {})
    found = [names[i] for i in sorted(names)]
    expected = list(config["classes"])
    if found != expected:
        raise SystemExit(
            f"Class order mismatch.\n"
            f"  dataset data.yaml: {found}\n"
            f"  config classes:    {expected}\n"
            "The label files encode the dataset's indices. Update the config "
            "to match the dataset, never the other way round."
        )

    counts = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        folder = root / data[split]
        counts[split] = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not counts.get("train"):
        raise SystemExit("No training images found")
    if not counts.get("test"):
        print("  WARNING: no test split - benchmark metrics will be unavailable")

    label_files = glob.glob(str(root / "train" / "labels" / "*.txt"))
    if not label_files:
        raise SystemExit("No label files found under train/labels")

    sample = Path(label_files[0]).read_text().split("\n")[0].split()
    if len(sample) == 5:
        raise SystemExit(
            "Labels look like detection boxes (5 values per row), not polygons. "
            "Re-export from Roboflow as an instance segmentation format."
        )
    vertices = (len(sample) - 1) // 2

    print(f"  dataset: {root}")
    print(f"  classes: {found} (+ background at index {config['background_index']})")
    print(f"  splits:  {counts}")
    print(f"  labels:  polygon format, {vertices} vertices in first row")

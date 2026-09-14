# Aerial Imagery Semantic Segmentation

Semantic segmentation of `road`, `divider` and `sidewalk` in aerial imagery.
Training runs on Colab; inference runs locally.

See `CLAUDE.md` for project context and `docs/DECISIONS.md` for the reasoning
behind the technical choices.

## Layout

```
.
├── configs/
│   ├── base.yaml          shared settings - change things here
│   ├── v0.1-100.yaml      run configs - only what differs
│   ├── v0.1-300.yaml
│   └── v0.1-500.yaml
├── src/aerial/
│   ├── config.py          base + run merge, CLI overrides, guards
│   ├── dataset.py         acquisition, data.yaml repair, validation
│   ├── train.py           training entry point
│   └── inference.py       local inference
└── notebooks/
    └── colab_train.ipynb
```

## Local setup

```bash
uv sync                    # inference only, CPU torch
uv sync --group data       # adds roboflow for dataset downloads
uv sync --all-groups       # everything
```

Inference:

```bash
uv run aerial-infer --path image.jpg --weights best.pt
```

## Training on Colab

Runtime → Change runtime type → GPU, then:

```python
from google.colab import drive, userdata
drive.mount('/content/drive')

!git clone https://github.com/YOU/aerial-segmentation.git
%cd aerial-segmentation
!pip install -q ultralytics roboflow pyyaml opencv-python
!pip install -q -e .

import os
os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')

!aerial-train --config configs/v0.1-100.yaml
```

Output lands in `/content/drive/MyDrive/aerial/v0.1-100/`:

```
v0.1-100/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── manifest.json         config, metrics, environment
└── *.png                 curves, confusion matrix
```

Because `project` points straight at Drive, a dropped session loses nothing.

## Running the three comparisons

```bash
aerial-train --config configs/v0.1-100.yaml
aerial-train --config configs/v0.1-300.yaml
aerial-train --config configs/v0.1-500.yaml
```

Only `train.epochs` differs between them. Everything else is inherited from
`base.yaml`, so the comparison isolates one variable. Resist the urge to
tweak a second setting mid-sequence - two variables at once makes the result
unattributable.

## Switching datasets

Edit the `dataset` block in `base.yaml`, or override per run:

```bash
# different Roboflow version
aerial-train --config configs/v0.1-100.yaml --set dataset.version=3

# an entirely different Roboflow project
aerial-train --config configs/v0.1-100.yaml \
  --set dataset.project=other-project --set dataset.version=1

# a local folder containing data.yaml
aerial-train --config configs/v0.1-100.yaml \
  --set dataset.source=local --set dataset.path=/content/mydata
```

Any config value can be overridden with `--set key.path=value`, which makes
one-off experiments cheap without editing a committed file.

Give a new dataset its own `run_name`, or the run will refuse to overwrite an
existing output folder.

## Validate before spending GPU hours

```bash
aerial-train --config configs/v0.1-100.yaml --dry-run
```

Downloads the dataset, repairs `data.yaml`, and checks class order, split
counts and label format. Takes seconds; saves a run that would have failed
twenty minutes in.

## Guards worth knowing about

- **Class order** must match `base.yaml`. Label files encode the dataset's
  indices, so update the config to match the dataset, never the reverse.
- **`val.imgsz` must equal `train.imgsz`.** Evaluating at a different
  resolution measures something other than what you trained.
- **A stray `masks/` folder** in a Roboflow export silently switches the
  loader to PNG-mask mode. `dataset.py` renames it.
- **`exist_ok` is False.** A second run with the same name fails rather than
  overwriting results you may still need.

## Reading the metrics

Semantic segmentation reports **IoU**, not mAP. The run prints per-class IoU,
pixel accuracy, pixel count and image count, plus two mIoU figures.

Quote **`miou_classes_only`**. Background is the complement of everything else,
sits near 0.95, and inflates any mean it enters.

`pixels` and `images` are the columns to read first - they tell you how much
evidence each score rests on. A class appearing in 4 of 20 test images has a
wide confidence interval whatever its IoU says.
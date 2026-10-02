#!/usr/bin/env python3
"""Train a segmentation-models-pytorch model on the aerial dataset.

    python train_smp.py --data ./dataset --epochs 300 --imgsz 1280

Deliberately matched to the YOLO26-sem runs so the comparison isolates
architecture: same splits, same 1280 resolution, same AdamW at 1e-3, same
cosine schedule, same CE+Dice loss combination, same per-class IoU
computed from a confusion matrix with background excluded from the mean.

Expects Roboflow's "Semantic Segmentation Masks" export:

    dataset/
      train/  <images>  + <stem>_mask.png
      valid/
      test/
      _classes.csv          (Roboflow's colour -> class mapping)

Run with --inspect first. Roboflow writes RGB colour-coded masks in some
exports and single-channel index masks in others, and feeding the wrong one
to the loss produces a model that trains happily and predicts nonsense.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Must match the YOLO dataset: Roboflow's alphabetical order, background last.
CLASSES = ["divider", "road", "sidewalk"]
BACKGROUND_INDEX = 3
NUM_CLASSES = len(CLASSES) + 1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class AerialDataset(Dataset):
    """Image + single-channel index mask.

    Masks are resized with INTER_NEAREST. Any interpolation that averages
    neighbouring pixels invents class indices that do not exist - a mask
    resized bilinearly between class 1 and class 3 produces class 2.
    """

    def __init__(self, root: Path, split: str, imgsz: int, augment: bool = False):
        self.imgsz = imgsz
        self.augment = augment
        self.pairs = self._index(Path(root) / split)
        if not self.pairs:
            raise SystemExit(f"No image/mask pairs found under {Path(root) / split}")

    @staticmethod
    def _index(folder: Path) -> list[tuple[Path, Path]]:
        pairs = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.stem.endswith("_mask"):
                continue
            mask = path.with_name(f"{path.stem}_mask.png")
            if mask.exists():
                pairs.append((path, mask))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        image_path, mask_path = self.pairs[i]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            raise RuntimeError(
                f"{mask_path} is a colour mask. Convert to single-channel "
                "index masks first - run with --inspect for details."
            )

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.rand() < 0.5:
            # Vertical flip only. Horizontal mirroring puts traffic on the
            # wrong side of the road, which the divider class encodes.
            image, mask = image[::-1].copy(), mask[::-1].copy()

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)
        return image, torch.from_numpy(mask).long()


def inspect(root: Path) -> None:
    """Report what the export actually contains before training on it."""
    root = Path(root)
    print(f"Inspecting {root}\n")
    for split in ("train", "valid", "test"):
        folder = root / split
        if not folder.is_dir():
            print(f"  {split}: MISSING")
            continue
        pairs = AerialDataset._index(folder)
        print(f"  {split}: {len(pairs)} image/mask pair(s)")
        if not pairs:
            continue
        mask = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print("    could not read first mask")
            continue
        if mask.ndim == 3:
            colours = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
            print(f"    RGB colour mask, shape {mask.shape}")
            print(f"    distinct colours: {colours.tolist()}")
            print("    -> needs conversion to index masks")
        else:
            values = np.unique(mask)
            print(f"    index mask, shape {mask.shape}, values {values.tolist()}")
            if values.max() >= NUM_CLASSES:
                print(
                    f"    WARNING: value {values.max()} exceeds num_classes-1 ({NUM_CLASSES - 1})"
                )

    csv = root / "_classes.csv"
    if csv.exists():
        print(f"\n_classes.csv:\n{csv.read_text().strip()}")
    else:
        print("\nNo _classes.csv found")


def colour_to_index(root: Path, mapping: dict[tuple, int]) -> int:
    """Rewrite RGB colour masks as single-channel index masks, in place."""
    converted = 0
    for split in ("train", "valid", "test"):
        folder = Path(root) / split
        if not folder.is_dir():
            continue
        for _, mask_path in AerialDataset._index(folder):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 3:
                continue
            rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            out = np.full(rgb.shape[:2], BACKGROUND_INDEX, np.uint8)
            for colour, index in mapping.items():
                out[np.all(rgb == np.array(colour), axis=-1)] = index
            cv2.imwrite(str(mask_path), out)
            converted += 1
    return converted


# ---------------------------------------------------------------------------
# Metrics - confusion matrix, matching how Ultralytics reports per-class IoU
# ---------------------------------------------------------------------------


def update_confusion(matrix: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> None:
    valid = truth < NUM_CLASSES
    idx = NUM_CLASSES * truth[valid].astype(int) + pred[valid].astype(int)
    matrix += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(matrix: np.ndarray) -> dict:
    """Per-class IoU = TP / (TP + FP + FN), read off the confusion matrix."""
    intersection = np.diag(matrix).astype(float)
    union = matrix.sum(1) + matrix.sum(0) - np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, np.nan)

    labels = [*CLASSES, "background"]
    per_class = {labels[i]: float(iou[i]) for i in range(NUM_CLASSES)}
    class_only = [per_class[c] for c in CLASSES if not np.isnan(per_class[c])]
    return {
        "per_class": per_class,
        # Background sits very high and would flatter the mean, exactly as in
        # the YOLO benchmark. Report classes only.
        "miou_classes": float(np.mean(class_only)) if class_only else float("nan"),
        "miou_with_bg": float(np.nanmean(iou)),
        "pixel_accuracy": float(np.diag(matrix).sum() / max(1, matrix.sum())),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, masks in loader:
        logits = model(images.to(device))
        pred = logits.argmax(1).cpu().numpy()
        update_confusion(matrix, pred.ravel(), masks.numpy().ravel())
    return iou_from_confusion(matrix)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_model(arch: str, encoder: str):
    import segmentation_models_pytorch as smp

    factory = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "segformer": getattr(smp, "Segformer", None),
    }.get(arch)
    if factory is None:
        raise SystemExit(f"Unknown or unavailable architecture: {arch}")

    return factory(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument(
        "--inspect", action="store_true", help="report the export's mask format, then exit"
    )
    p.add_argument(
        "--arch",
        default="unet",
        choices=["unet", "unetplusplus", "deeplabv3plus", "fpn", "segformer"],
    )
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("runs/smp"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.inspect:
        inspect(args.data)
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train = DataLoader(
        AerialDataset(args.data, "train", args.imgsz, augment=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val = DataLoader(
        AerialDataset(args.data, "valid", args.imgsz),
        batch_size=max(1, args.batch // 2),
        num_workers=args.workers,
    )
    test = DataLoader(
        AerialDataset(args.data, "test", args.imgsz),
        batch_size=max(1, args.batch // 2),
        num_workers=args.workers,
    )
    print(f"train {len(train.dataset)}  val {len(val.dataset)}  test {len(test.dataset)}")

    import segmentation_models_pytorch as smp

    model = build_model(args.arch, args.encoder).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}/{args.encoder}: {params:,} parameters")

    # YOLO26-sem optimises cross-entropy plus dice. Matching the loss matters
    # as much as matching the schedule - a different objective is a different
    # experiment.
    ce = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler(device)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, started = 0.0, time.time()
        for images, masks in train:
            images, masks = images.to(device, non_blocking=True), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                logits = model(images)
                loss = ce(logits, masks) + dice(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += loss.item()
        schedule.step()

        metrics = evaluate(model, val, device)
        history.append({"epoch": epoch, "loss": total / len(train), **metrics})
        print(
            f"{epoch:4d}/{args.epochs}  loss {total / len(train):.4f}  "
            f"mIoU {metrics['miou_classes']:.4f}  {time.time() - started:.0f}s"
        )

        if metrics["miou_classes"] > best:
            best = metrics["miou_classes"]
            torch.save(model.state_dict(), args.out / "best.pt")
        torch.save(model.state_dict(), args.out / "last.pt")

    model.load_state_dict(torch.load(args.out / "best.pt"))
    final = evaluate(model, test, device)

    print(f"\n{'class':14s} {'IoU':>8s}")
    for name, value in final["per_class"].items():
        print(f"{name:14s} {value:8.4f}")
    print(f"\nmIoU (classes only): {final['miou_classes']:.4f}")

    (args.out / "results.json").write_text(
        json.dumps(
            {"config": vars(args), "test": final, "history": history},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

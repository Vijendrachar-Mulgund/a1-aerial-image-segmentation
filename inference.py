#!/usr/bin/env python3
"""Run the semantic segmentation model over an image or a folder of images.

    python inference.py --path road.jpg
    python inference.py --path ./images --output ./results
    python inference.py --path road.jpg --weights runs/v1/weights/best.pt

Writes three files per input, into a folder next to this script by default:

    <stem>_mask.png      single channel, pixel value = class index
    <stem>_overlay.png   original with colour-coded classes blended over it
    <stem>.json          polygon outlines in image pixel coordinates

The mask is the raw model output. The overlay is rendered *from the
polygons*, not from the mask, so that what the client sees and what the JSON
says cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Class order comes from data.yaml and is authoritative. Background is the
# channel Ultralytics appends for the polygon path; it is never drawn.
CLASS_NAMES = {0: "divider", 1: "road", 2: "sidewalk", 3: "vehicle", 4: "background"}

# BGR. Chosen to sit clear of road grey and vegetation green.
CLASS_COLOURS = {
    0: (0, 140, 255),  # divider  — orange
    1: (255, 80, 0),  # road     — blue
    2: (200, 0, 200),  # sidewalk — magenta
    3: (60, 200, 60),  # vehicle  — green
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

MIN_REGION_AREA = 200  # px; drops speckle
SIMPLIFY_EPSILON = 2.0  # px; Douglas-Peucker tolerance
OVERLAY_ALPHA = 0.45


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", required=True, type=Path, help="image file or folder of images")
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "output",
        help="output folder (default: ./output beside this script)",
    )
    parser.add_argument("--weights", type=Path, default=here / "best.pt", help="model weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="must match training")
    parser.add_argument("--device", default=None, help="cuda device, or 'cpu'")
    parser.add_argument("--no-overlay", action="store_true", help="skip the overlay image")
    parser.add_argument(
        "--class-map",
        default=None,
        help='render a different class set, e.g. "0=road,1=sidewalk" for a '
        "Cityscapes-pretrained baseline. Indices are the MODEL's own class "
        "indices; names are what appears in the legend. Omit to use the "
        "fine-tuned defaults.",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="print the model's class names and exit",
    )
    return parser.parse_args()


def apply_class_map(spec: str) -> None:
    """Replace the module-level class table.

    A model's class indices are its own. Comparing a Cityscapes-pretrained
    baseline against a fine-tuned model means rendering different indices
    under the same names, so the tables have to be swappable rather than
    hardcoded.
    """
    global CLASS_NAMES, CLASS_COLOURS
    palette = [
        (0, 140, 255),
        (255, 80, 0),
        (200, 0, 200),
        (60, 200, 60),
        (0, 215, 255),
        (200, 120, 0),
    ]
    names, colours = {}, {}
    for i, item in enumerate(spec.split(",")):
        if "=" not in item:
            raise SystemExit(f"--class-map entry must be index=name, got: {item}")
        idx, name = item.split("=", 1)
        idx = int(idx.strip())
        names[idx] = name.strip()
        colours[idx] = palette[i % len(palette)]
    CLASS_NAMES = names
    CLASS_COLOURS = colours


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            raise SystemExit(f"No images found under {path}")
        return found
    raise SystemExit(f"Path does not exist: {path}")


def mask_from_result(result, height: int, width: int) -> np.ndarray:
    """Pull a single-channel class-index mask out of an Ultralytics result.

    The semantic task exposes its prediction in different places depending on
    version, so probe rather than assume, and fail with a readable message
    instead of an obscure attribute error.
    """
    for attr in ("semantic_mask", "semantic", "masks", "probs"):
        obj = getattr(result, attr, None)
        if obj is None:
            continue
        data = getattr(obj, "data", obj)
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)

        if arr.ndim == 3 and arr.shape[0] > 1:  # (C, H, W) logits
            mask = arr.argmax(axis=0)
        elif arr.ndim == 3:  # (1, H, W)
            mask = arr[0]
        elif arr.ndim == 2:  # (H, W) already indices
            mask = arr
        else:
            continue

        mask = mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    raise RuntimeError(
        "Could not read a segmentation mask from the model result. "
        f"Available attributes: {[a for a in dir(result) if not a.startswith('_')]}"
    )


def polygons_for_class(mask: np.ndarray, class_id: int) -> list[dict]:
    """Contour, simplify and package one class's regions."""
    binary = (mask == class_id).astype(np.uint8)
    if not binary.any():
        return []

    # Close first, then open. Sidewalks fragment under tree canopy and
    # dividers under building shadow; closing rejoins them before the
    # opening pass removes genuine speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # RETR_CCOMP keeps hierarchy: a divider inside a road is a hole in the
    # road polygon, not a separate region.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    regions = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] != -1:  # inner ring; attached below
            continue
        if cv2.contourArea(contour) < MIN_REGION_AREA:
            continue

        exterior = cv2.approxPolyDP(contour, SIMPLIFY_EPSILON, True)
        if len(exterior) < 3:
            continue

        holes = []
        child = hierarchy[idx][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= MIN_REGION_AREA:
                ring = cv2.approxPolyDP(contours[child], SIMPLIFY_EPSILON, True)
                if len(ring) >= 3:
                    holes.append(ring.reshape(-1, 2).tolist())
            child = hierarchy[child][0]

        regions.append(
            {
                "class_id": class_id,
                "class": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "area_px": int(cv2.contourArea(contour)),
                "exterior": exterior.reshape(-1, 2).tolist(),
                "holes": holes,
            }
        )

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions


def draw_legend(image: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Legend for the classes actually present, with coverage percentages.

    Colour without a key is meaningless to anyone who did not write the
    script. The coverage figure doubles as a sanity check: a class claiming
    30% of an aerial frame is almost certainly over-predicting.
    """
    present = []
    for cid, name in CLASS_NAMES.items():
        if name == "background" or cid not in CLASS_COLOURS:
            continue
        area = sum(r["area_px"] for r in regions if r["class_id"] == cid)
        if area > 0:
            count = sum(1 for r in regions if r["class_id"] == cid)
            present.append((cid, name, area, count))
    if not present:
        return image

    h, w = image.shape[:2]
    s = max(1.0, min(w, h) / 900.0)
    pad, swatch, row = int(14 * s), int(22 * s), int(34 * s)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, max(1, int(1.6 * s))

    labels = [f"{n}  -  {c} region(s), {a / (w * h) * 100:.1f}%" for _, n, a, c in present]
    tw = max(cv2.getTextSize(t, font, fs, ft)[0][0] for t in labels)
    bw = pad * 2 + swatch + int(10 * s) + tw
    bh = pad * 2 + row * len(present)
    x0, y0 = pad, pad

    panel = image.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), -1)
    image = cv2.addWeighted(panel, 0.82, image, 0.18, 0)
    cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), max(1, int(1.5 * s)))

    for i, ((cid, _, _, _), text) in enumerate(zip(present, labels, strict=False)):
        ty = y0 + pad + i * row
        cv2.rectangle(
            image, (x0 + pad, ty), (x0 + pad + swatch, ty + swatch), CLASS_COLOURS[cid], -1
        )
        cv2.rectangle(image, (x0 + pad, ty), (x0 + pad + swatch, ty + swatch), (60, 60, 60), 1)
        cv2.putText(
            image,
            text,
            (x0 + pad + swatch + int(10 * s), ty + int(swatch * 0.8)),
            font,
            fs,
            (20, 20, 20),
            ft,
            cv2.LINE_AA,
        )
    return image


def render_overlay(image: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Draw filled polygons, then outline them, then blend.

    Rendered from the polygons rather than the mask so the picture always
    agrees with the JSON.
    """
    fill = image.copy()
    # Largest first so smaller classes land on top.
    for region in sorted(regions, key=lambda r: r["area_px"], reverse=True):
        colour = CLASS_COLOURS.get(region["class_id"])
        if colour is None:
            continue
        exterior = np.array(region["exterior"], dtype=np.int32)
        holes = [np.array(h, dtype=np.int32) for h in region["holes"]]
        cv2.fillPoly(fill, [exterior], colour)
        if holes:
            cv2.fillPoly(fill, holes, (0, 0, 0))  # punch out, restored below

    blended = cv2.addWeighted(fill, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0)

    for region in regions:
        colour = CLASS_COLOURS.get(region["class_id"])
        if colour is None:
            continue
        cv2.polylines(
            blended,
            [np.array(region["exterior"], dtype=np.int32)],
            True,
            colour,
            2,
        )
        for hole in region["holes"]:
            cv2.polylines(blended, [np.array(hole, dtype=np.int32)], True, colour, 2)
    return draw_legend(blended, regions)


def process(image_path: Path, model, args, out_dir: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    predict_kwargs = {"imgsz": args.imgsz, "verbose": False}
    if args.device is not None:
        predict_kwargs["device"] = args.device
    result = model.predict(str(image_path), **predict_kwargs)[0]

    mask = mask_from_result(result, height, width)

    regions: list[dict] = []
    for class_id in CLASS_NAMES:
        if CLASS_NAMES[class_id] == "background":
            continue
        regions.extend(polygons_for_class(mask, class_id))

    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)

    if not args.no_overlay:
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), render_overlay(image, regions))

    payload = {
        "image": image_path.name,
        "width": width,
        "height": height,
        "classes": {k: v for k, v in CLASS_NAMES.items() if v != "background"},
        "coordinate_space": "image pixels, origin top-left",
        "regions": regions,
        "summary": {
            name: {
                "regions": sum(1 for r in regions if r["class"] == name),
                "area_px": sum(r["area_px"] for r in regions if r["class"] == name),
            }
            for name in CLASS_NAMES.values()
            if name != "background"
        },
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    return payload["summary"]


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        print("Pass --weights, or place best.pt beside this script.", file=sys.stderr)
        return 1

    inputs = collect_inputs(args.path)
    args.output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    model_names = getattr(model, "names", None) or getattr(model.model, "names", {})
    if args.list_classes:
        for i in sorted(model_names):
            print(f"{i}: {model_names[i]}")
        return 0

    if args.class_map:
        apply_class_map(args.class_map)

    unknown = [i for i in CLASS_NAMES if model_names and i not in model_names]
    if unknown:
        print(
            f"WARNING: indices {unknown} are not in this model "
            f"({len(model_names)} classes). Use --list-classes.",
            file=sys.stderr,
        )
    for i, name in CLASS_NAMES.items():
        if model_names and i in model_names and model_names[i] != name:
            print(f"  note: rendering model class {i} ('{model_names[i]}') as '{name}'")

    failures = 0
    for image_path in inputs:
        try:
            summary = process(image_path, model, args, args.output)
            parts = " ".join(
                f"{name}={s['regions']}r/{s['area_px']:,}px" for name, s in summary.items()
            )
            print(f"{image_path.name}: {parts}")
        except Exception as exc:  # keep going through a batch
            failures += 1
            print(f"{image_path.name}: FAILED — {exc}", file=sys.stderr)

    print(f"\n{len(inputs) - failures}/{len(inputs)} written to {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

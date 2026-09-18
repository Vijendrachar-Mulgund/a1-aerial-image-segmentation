#!/usr/bin/env python3
"""Build side-by-side comparison sheets from several models' overlay output.

    python compare_outputs.py --root ./compare
    python compare_outputs.py --root ./compare --original ./test_images
    python compare_outputs.py --root ./compare --cols 3 --suffix _overlay.png

Expects one subfolder per model, each holding overlays with matching
filenames:

    compare/
      v0.1-100-s/road_001_overlay.png
      v0.2-100-s/road_001_overlay.png
      v0.2-300-s/road_001_overlay.png
      ...

Writes one sheet per source image into --output, each panel labelled with
the model that produced it. Subfolder order is preserved as given on the
command line, or sorted alphabetically otherwise, so every sheet places the
same model in the same position - which is what makes them scannable in
bulk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

LABEL_BAND = 46  # px, scaled with panel size
GUTTER = 8
BG = (248, 248, 248)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root", required=True, type=Path, help="folder containing one subfolder per model"
    )
    p.add_argument("--output", type=Path, default=here / "output-comparison")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subfolder names, in the order you want them laid out. "
        "Defaults to all subfolders, sorted.",
    )
    p.add_argument(
        "--original",
        type=Path,
        default=None,
        help="folder of source images, added as the first panel",
    )
    p.add_argument(
        "--suffix",
        default="_overlay.png",
        help="overlay filename suffix to strip when matching (default: _overlay.png)",
    )
    p.add_argument(
        "--cols", type=int, default=0, help="panels per row. 0 picks a near-square layout."
    )
    p.add_argument(
        "--width", type=int, default=900, help="width each panel is resized to (default: 900)"
    )
    p.add_argument(
        "--limit", type=int, default=0, help="only build this many sheets, for a quick look"
    )
    return p.parse_args()


def discover_models(root: Path, given: list[str] | None) -> list[Path]:
    if given:
        folders = [root / name for name in given]
        missing = [f for f in folders if not f.is_dir()]
        if missing:
            raise SystemExit("Not found under --root: " + ", ".join(m.name for m in missing))
        return folders
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f"No subfolders under {root}")
    return folders


def index_folder(folder: Path, suffix: str) -> dict[str, Path]:
    """Map a stable key to each overlay path.

    The key strips the overlay suffix so `road_001_overlay.png` matches
    `road_001.jpg` in the originals folder.
    """
    out = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = path.name
        key = name[: -len(suffix)] if suffix and name.endswith(suffix) else path.stem
        out[key] = path
    return out


def label_panel(image: np.ndarray, text: str, width: int) -> np.ndarray:
    """Resize to a common width and add a caption band underneath."""
    h, w = image.shape[:2]
    scale = width / w
    panel = cv2.resize(image, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    s = max(1.0, width / 900.0)
    band = int(LABEL_BAND * s)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, max(1, int(1.8 * s))

    strip = np.full((band, width, 3), 255, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
    cv2.putText(
        strip, text, ((width - tw) // 2, (band + th) // 2), font, fs, (25, 25, 25), ft, cv2.LINE_AA
    )
    cv2.line(strip, (0, 0), (width, 0), (200, 200, 200), 1)

    return np.vstack([panel, strip])


def build_sheet(panels: list[np.ndarray], cols: int) -> np.ndarray:
    """Lay panels out on a grid, padding uneven rows with blanks.

    Panels share a width but may differ in height if the source images had
    different aspect ratios, so each row is padded to its tallest member.
    """
    if not panels:
        raise ValueError("no panels")
    width = panels[0].shape[1]
    rows = [panels[i : i + cols] for i in range(0, len(panels), cols)]

    built = []
    for row in rows:
        while len(row) < cols:
            row.append(np.full((row[0].shape[0], width, 3), BG, np.uint8))
        tallest = max(p.shape[0] for p in row)
        padded = []
        for p in row:
            if p.shape[0] < tallest:
                pad = np.full((tallest - p.shape[0], width, 3), BG, np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
            padded.append(np.full((tallest, GUTTER, 3), BG, np.uint8))
        built.append(np.hstack(padded[:-1]))

    total_width = max(r.shape[1] for r in built)
    stacked = []
    for r in built:
        if r.shape[1] < total_width:
            pad = np.full((r.shape[0], total_width - r.shape[1], 3), BG, np.uint8)
            r = np.hstack([r, pad])
        stacked.append(r)
        stacked.append(np.full((GUTTER, total_width, 3), BG, np.uint8))

    return np.vstack(stacked[:-1])


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"--root is not a folder: {args.root}")

    folders = discover_models(args.root, args.models)
    indexes = {f.name: index_folder(f, args.suffix) for f in folders}

    originals = {}
    if args.original:
        if not args.original.is_dir():
            raise SystemExit(f"--original is not a folder: {args.original}")
        originals = {
            p.stem: p for p in sorted(args.original.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES
        }

    # Only build sheets for images every model produced, so a sheet never
    # silently omits a model and invites a wrong comparison.
    keys = set.intersection(*(set(idx) for idx in indexes.values()))
    if not keys:
        print("No filenames common to every model folder.", file=sys.stderr)
        for name, idx in indexes.items():
            sample = sorted(idx)[:3]
            print(f"  {name}: {len(idx)} files, e.g. {sample}", file=sys.stderr)
        print("\nCheck --suffix matches your overlay filenames.", file=sys.stderr)
        return 1

    dropped = {name: sorted(set(idx) - keys) for name, idx in indexes.items()}
    for name, missing in dropped.items():
        if missing:
            print(f"  note: {name} has {len(missing)} image(s) no other model produced")

    panel_count = len(folders) + (1 if originals else 0)
    cols = args.cols or int(np.ceil(np.sqrt(panel_count)))

    args.output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(keys)
    if args.limit:
        ordered = ordered[: args.limit]

    written = 0
    for key in ordered:
        panels = []

        if originals:
            src = originals.get(key)
            if src is not None:
                image = cv2.imread(str(src))
                if image is not None:
                    panels.append(label_panel(image, "original", args.width))

        for folder in folders:
            path = indexes[folder.name][key]
            image = cv2.imread(str(path))
            if image is None:
                print(f"  skip unreadable: {path}", file=sys.stderr)
                continue
            panels.append(label_panel(image, folder.name, args.width))

        if not panels:
            continue

        sheet = build_sheet(panels, cols)
        out_path = args.output / f"{key}_comparison.png"
        cv2.imwrite(str(out_path), sheet)
        written += 1
        print(f"{out_path.name}  ({sheet.shape[1]}x{sheet.shape[0]})")

    print(f"\n{written} sheet(s) written to {args.output}")
    print(f"Layout: {cols} column(s), {panel_count} panel(s) per sheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())

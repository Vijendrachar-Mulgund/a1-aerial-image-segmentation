#!/usr/bin/env python3
"""Batch image conversions.

    uv run convert --greyscale path/to/folder
    uv run convert --rotate path/to/folder
    uv run convert --greyscale --rotate path/to/folder

Output goes to `output-grayscale/` and `output-rotation/` beside this
script, created on demand. A single file may be passed instead of a folder.

Note on annotations: rotating an image invalidates its polygon labels unless
they are rotated to match. Use --labels to transform matching .txt files
alongside the images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

GREYSCALE_DIR = "output-grayscale"
ROTATION_DIR = "output-rotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--greyscale",
        "--grayscale",
        dest="greyscale",
        action="store_true",
        help=f"convert to greyscale, written to {GREYSCALE_DIR}/",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help=f"rotate 180 degrees, written to {ROTATION_DIR}/",
    )
    parser.add_argument("path", type=Path, help="image file or folder of images")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="parent folder for the output directories (default: beside this script)",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="also transform matching YOLO polygon .txt labels "
        "(rotation only; greyscale leaves geometry unchanged)",
    )
    parser.add_argument(
        "--three-channel",
        action="store_true",
        help="write greyscale back as 3-channel BGR, which is what the model expects as input",
    )

    args = parser.parse_args()
    if not (args.greyscale or args.rotate):
        parser.error("choose at least one of --greyscale or --rotate")
    return args


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            raise SystemExit(f"No images found under {path}")
        return found
    raise SystemExit(f"Path does not exist: {path}")


def to_greyscale(image, three_channel: bool = False):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR) if three_channel else grey


def rotate_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)


def rotate_labels_180(text: str) -> str:
    """Rotate normalised YOLO polygon coordinates by 180 degrees.

    Coordinates are normalised to 0-1, so a 180 degree rotation about the
    image centre is simply x -> 1-x, y -> 1-y. Class index is untouched.
    Vertex winding reverses under this transform, which OpenCV and the
    Ultralytics loader both tolerate.
    """
    out_lines = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        class_id, coords = parts[0], [float(v) for v in parts[1:]]
        if len(coords) % 2:
            raise ValueError(f"Odd coordinate count in line: {line[:60]}")
        flipped = [
            f"{1.0 - v:.6f}" if i % 2 == 0 else f"{1.0 - v:.6f}" for i, v in enumerate(coords)
        ]
        out_lines.append(" ".join([class_id, *flipped]))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def find_label(image_path: Path) -> Path | None:
    """Look for a matching .txt beside the image, or in a sibling labels/."""
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
    ]
    return next((c for c in candidates if c.exists()), None)


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args.path)
    parent = args.output or Path(__file__).resolve().parent

    targets = []
    if args.greyscale:
        targets.append(("greyscale", parent / GREYSCALE_DIR))
    if args.rotate:
        targets.append(("rotate", parent / ROTATION_DIR))
    for _, directory in targets:
        directory.mkdir(parents=True, exist_ok=True)

    written = failed = labels_written = 0

    for image_path in inputs:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path.name}: FAILED - could not read", file=sys.stderr)
            failed += 1
            continue

        for operation, directory in targets:
            try:
                if operation == "greyscale":
                    result = to_greyscale(image, args.three_channel)
                else:
                    result = rotate_180(image)

                out_path = directory / image_path.name
                if not cv2.imwrite(str(out_path), result):
                    raise RuntimeError(f"write failed: {out_path}")
                written += 1

                if args.labels:
                    label = find_label(image_path)
                    if label is not None:
                        text = label.read_text()
                        if operation == "rotate":
                            text = rotate_labels_180(text)
                        (directory / f"{image_path.stem}.txt").write_text(text)
                        labels_written += 1

            except Exception as exc:
                print(f"{image_path.name} [{operation}]: FAILED - {exc}", file=sys.stderr)
                failed += 1

        print(f"{image_path.name}: {', '.join(op for op, _ in targets)}")

    print(f"\n{written} image(s) written across {len(targets)} folder(s)")
    if args.labels:
        print(f"{labels_written} label file(s) written")
    if failed:
        print(f"{failed} failure(s)", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

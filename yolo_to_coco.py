"""Utilities for converting YOLO annotations to a COCO-style JSON file."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class CocoImage:
    image_id: int
    path: Path
    width: int
    height: int
    rel_path: str


def _load_classes(classes_file: Path) -> List[str]:
    with classes_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _iter_image_files(images_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                yield path
    else:
        for path in images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                yield path


def _load_image_metadata(images_dir: Path, recursive: bool) -> List[CocoImage]:
    images: List[CocoImage] = []
    image_id = 1

    for file_path in sorted(_iter_image_files(images_dir, recursive)):
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except OSError:
            print(f"Warning: failed to open image: {file_path}")
            continue

        rel_path = os.path.relpath(file_path, images_dir)
        images.append(
            CocoImage(
                image_id=image_id,
                path=file_path,
                width=width,
                height=height,
                rel_path=rel_path.replace(os.sep, "/"),
            )
        )
        image_id += 1

    return images


def yolo_to_coco(
    yolo_root: Path,
    output_json: Path,
    *,
    recursive: bool = True,
    classes_file: Optional[Path] = None,
) -> None:
    """Convert a YOLO dataset rooted at *yolo_root* to COCO JSON."""

    yolo_root = Path(yolo_root).resolve()
    output_json = Path(output_json)

    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    if classes_file is None:
        classes_file = yolo_root / "classes.txt"

    categories: List[Dict[str, object]]
    class_index_to_id: Dict[int, int] = {}

    if classes_file.exists():
        class_names = _load_classes(classes_file)
        categories = [
            {"id": idx + 1, "name": name} for idx, name in enumerate(class_names)
        ]
        class_index_to_id = {idx: idx + 1 for idx in range(len(class_names))}
    else:
        class_names = []
        categories = []

    images = _load_image_metadata(images_dir, recursive)
    annotations: List[Dict[str, object]] = []
    next_ann_id = 1

    for image in images:
        label_rel = os.path.splitext(image.rel_path)[0] + ".txt"
        label_path = labels_dir / label_rel

        if not label_path.exists():
            continue

        try:
            with label_path.open("r", encoding="utf-8") as label_file:
                lines = [line.strip() for line in label_file if line.strip()]
        except OSError:
            print(f"Warning: failed to read label file: {label_path}")
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                print(f"Warning: malformed annotation in {label_path}: {line}")
                continue

            try:
                class_idx = int(parts[0])
                x_center = float(parts[1]) * image.width
                y_center = float(parts[2]) * image.height
                bbox_width = float(parts[3]) * image.width
                bbox_height = float(parts[4]) * image.height
            except ValueError:
                print(f"Warning: non-numeric annotation in {label_path}: {line}")
                continue

            if class_idx < 0:
                print(f"Warning: negative class id in {label_path}: {line}")
                continue

            if class_names and class_idx >= len(class_names):
                print(
                    "Warning: class index exceeds classes list; skipping line in "
                    f"{label_path}: {line}"
                )
                continue

            category_id = class_index_to_id.get(class_idx)
            if category_id is None:
                category_id = len(categories) + 1
                categories.append({"id": category_id, "name": str(class_idx)})
                class_index_to_id[class_idx] = category_id

            x_min = x_center - bbox_width / 2.0
            y_min = y_center - bbox_height / 2.0

            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image.image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(bbox_width, 2),
                        round(bbox_height, 2),
                    ],
                    "area": round(bbox_width * bbox_height, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            next_ann_id += 1

    coco_dict = {
        "info": {
            "description": "YOLO to COCO conversion",
            "version": "1.0",
        },
        "licenses": [],
        "images": [
            {
                "id": image.image_id,
                "file_name": image.rel_path,
                "width": image.width,
                "height": image.height,
            }
            for image in images
        ],
        "annotations": annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(coco_dict, output_file, ensure_ascii=False, indent=2)

    print(
        f"COCO annotations saved to {output_json}. "
        f"Images: {len(images)}, annotations: {len(annotations)}, categories: {len(categories)}"
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO dataset (images/ + labels/) to COCO JSON",
    )
    parser.add_argument(
        "root",
        help="Path to YOLO dataset root (must contain images/ and labels/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="coco_annotations.json",
        help="Output COCO JSON path",
    )
    parser.add_argument(
        "--classes",
        help="Optional explicit path to classes.txt (defaults to root/classes.txt)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        default=True,
        action="store_false",
        help="Only scan direct children of images/",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    root = Path(args.root)
    classes_arg = Path(args.classes) if args.classes else None
    yolo_to_coco(
        root, Path(args.output), recursive=args.recursive, classes_file=classes_arg
    )


if __name__ == "__main__":
    main()

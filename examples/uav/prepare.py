"""Prepare the UAV normal dataset as fixed-size YOLO images with a video-isolated split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean

import cv2
import yaml

IMAGE_SIZE = (640, 352)
SEED = 0
CANDIDATE_COUNT = 10_000
SPLITS = ("train", "val", "test")
RATIOS = (Fraction(8, 10), Fraction(1, 10), Fraction(1, 10))
OUTPUT_NAME = "uav_normal_640x352_seed0"
EXPECTED_ROWS = 4_110
EXPECTED_GROUPS = 33
VIDEO_FRAME = re.compile(
    r"(?P<video>.+\.(?:mp4|mov|avi|mkv|wmv|flv|webm|mpeg|mpg|m4v))_\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE
)
DEFAULT_SOURCE = Path(__file__).resolve().parents[2].parent / "dataset/data/uav/normal"


def _source_path(root: Path, value: str) -> Path:
    """Resolve a CSV path and reject paths that escape the source directory."""
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        raise ValueError(f"CSV paths must be relative to the source directory: {value}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(f"CSV path escapes the source directory: {value}") from e
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_records(source: Path) -> list[dict]:
    """Read and validate source rows, labels, video names, and actual image hashes."""
    source = source.resolve()
    mapping = source / "source_mapping.csv"
    if not source.is_dir() or not mapping.is_file():
        raise FileNotFoundError(f"Expected source_mapping.csv in {source}")
    required = {"id", "image", "label", "source_image", "image_sha256", "boxes"}
    with mapping.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"source_mapping.csv must contain: {', '.join(sorted(required))}")
        rows = list(reader)

    records, ids = [], set()
    for row in rows:
        image_id = row["id"]
        if (
            not image_id
            or Path(image_id).name != image_id
            or image_id in (".", "..")
            or "\\" in image_id
            or image_id in ids
        ):
            raise ValueError(f"Invalid or duplicate image id: {image_id!r}")
        ids.add(image_id)
        image_path = _source_path(source, row["image"])
        label_path = _source_path(source, row["label"])
        video_name = Path(row["source_image"].replace("\\", "/")).name
        match = VIDEO_FRAME.fullmatch(video_name)
        if not match:
            raise ValueError(f"Could not identify video and frame from source_image: {row['source_image']}")
        video_group = match.group("video")

        label = label_path.read_text(encoding="utf-8")
        boxes = []
        for line_number, line in enumerate(label.splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Expected 5 YOLO fields in {label_path}:{line_number}")
            try:
                class_id = int(values[0])
                x, y, width, height = (float(value) for value in values[1:])
            except ValueError as e:
                raise ValueError(f"Invalid YOLO label in {label_path}:{line_number}") from e
            if class_id != 0 or not all(map(math.isfinite, (x, y, width, height))) or width <= 0 or height <= 0:
                raise ValueError(f"Expected finite class-0 boxes with positive size in {label_path}:{line_number}")
            boxes.append((width, height))
        try:
            declared_boxes = int(row["boxes"])
        except ValueError as e:
            raise ValueError(f"Invalid box count in source_mapping.csv for {image_id}") from e
        if declared_boxes != len(boxes):
            raise ValueError(f"Box count mismatch for {image_id}: CSV={declared_boxes}, label={len(boxes)}")

        actual_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        records.append(
            {
                "id": image_id,
                "image_path": image_path,
                "label_path": label_path,
                "source_image": row["source_image"],
                "video_group": video_group,
                "sha256": actual_sha256,
                "csv_sha256_matches": (row["image_sha256"] or "").lower() == actual_sha256,
                "boxes": boxes,
            }
        )

    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} images, found {len(records)}")
    groups = {record["video_group"] for record in records}
    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(f"Expected {EXPECTED_GROUPS} video groups, found {len(groups)}")
    return sorted(records, key=lambda record: record["id"])


def _select_split(records: list[dict]) -> tuple[dict[str, str], int, Fraction]:
    """Choose the best seeded group assignment with no class or actual-hash leakage."""
    by_group = defaultdict(list)
    sha_groups = defaultdict(set)
    for record in records:
        by_group[record["video_group"]].append(record)
        sha_groups[record["sha256"]].add(record["video_group"])
    groups = sorted(by_group)
    group_counts = {
        group: (
            len(by_group[group]),
            sum(bool(record["boxes"]) for record in by_group[group]),
            sum(not record["boxes"] for record in by_group[group]),
        )
        for group in groups
    }
    group_index = {group: index for index, group in enumerate(groups)}
    duplicate_group_indexes = [
        tuple(group_index[group] for group in values) for values in sha_groups.values() if len(values) > 1
    ]
    rng = random.Random(SEED)
    best_key, valid_candidates = None, 0

    for _ in range(CANDIDATE_COUNT):
        assignment = tuple(rng.choices(range(3), weights=(8, 1, 1), k=len(groups)))
        if any(len({assignment[index] for index in indexes}) > 1 for indexes in duplicate_group_indexes):
            continue
        counts = [[0, 0, 0] for _ in SPLITS]
        for index, group in enumerate(groups):
            split = assignment[index]
            for metric, count in enumerate(group_counts[group]):
                counts[split][metric] += count
        if any(not positive or not negative for _, positive, negative in counts):
            continue
        valid_candidates += 1
        error = sum(
            abs(Fraction(counts[split][metric], sum(row[metric] for row in counts)) - RATIOS[split])
            for metric in range(3)
            for split in range(3)
        )
        key = (error, assignment)
        if best_key is None or key < best_key:
            best_key = key

    if best_key is None:
        raise ValueError("No valid split found among 10,000 candidates; each split needs positive and negative images")
    return dict(zip(groups, (SPLITS[index] for index in best_key[1]))), valid_candidates, best_key[0]


def _summary(records: list[dict]) -> dict:
    """Summarize image, class, and resized-pixel box counts."""
    widths = [width * IMAGE_SIZE[0] for record in records for width, _ in record["boxes"]]
    heights = [height * IMAGE_SIZE[1] for record in records for _, height in record["boxes"]]
    return {
        "images": len(records),
        "positive_images": sum(bool(record["boxes"]) for record in records),
        "negative_images": sum(not record["boxes"] for record in records),
        "boxes": len(widths),
        "box_width_px": {
            "min": min(widths) if widths else None,
            "mean": mean(widths) if widths else None,
            "max": max(widths) if widths else None,
        },
        "box_height_px": {
            "min": min(heights) if heights else None,
            "mean": mean(heights) if heights else None,
            "max": max(heights) if heights else None,
        },
        "boxes_under_4px_both": sum(width < 4 and height < 4 for width, height in zip(widths, heights)),
    }


def prepare_dataset(source: Path, output: Path) -> dict:
    """Resize the full source dataset and write its fixed train/val/test split."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source dataset")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    records = _load_records(source)
    group_split, valid_candidates, error = _select_split(records)
    sha_splits = defaultdict(set)
    for record in records:
        record["split"] = group_split[record["video_group"]]
        sha_splits[record["sha256"]].add(record["split"])
    if any(len(splits) > 1 for splits in sha_splits.values()):
        raise ValueError("Identical source images would be split across datasets")

    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode source image: {record['image_path']}")
        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        destination = output / "images" / record["split"] / f"{record['id']}.png"
        if not cv2.imwrite(str(destination), resized):
            raise OSError(f"Could not write PNG image: {destination}")
        shutil.copyfile(record["label_path"], output / "labels" / record["split"] / f"{record['id']}.txt")

    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "uav"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = {
        "parameters": {
            "seed": SEED,
            "candidate_count": CANDIDATE_COUNT,
            "target_ratios": dict(zip(SPLITS, (float(ratio) for ratio in RATIOS))),
            "grouping": "source_image video basename; archive directories ignored",
            "selection_error": "sum of absolute proportion errors for images and positive/negative images",
            "image_size": list(IMAGE_SIZE),
            "resize_interpolation": "cv2.INTER_LINEAR",
        },
        "selected_error": {"numerator": error.numerator, "denominator": error.denominator, "value": float(error)},
        "valid_candidates": valid_candidates,
        "group_splits": group_split,
        "images": [
            {
                "id": record["id"],
                "split": record["split"],
                "video_group": record["video_group"],
                "source_image": record["source_image"],
                "image_sha256": record["sha256"],
                "csv_sha256_matches": record["csv_sha256_matches"],
                "positive": bool(record["boxes"]),
                "box_count": len(record["boxes"]),
            }
            for record in records
        ],
    }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    by_sha, by_group = defaultdict(list), defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)
        by_group[record["video_group"]].append(record)
    stats = {
        "source": str(source),
        "output": str(output),
        "parameters": manifest["parameters"],
        "selected_error": manifest["selected_error"],
        "valid_candidates": valid_candidates,
        "csv_sha256_mismatches": sum(not record["csv_sha256_matches"] for record in records),
        "overall": _summary(records),
        "splits": {split: _summary([r for r in records if r["split"] == split]) for split in SPLITS},
        "video_groups": [
            {
                "video_group": group,
                "split": group_split[group],
                "sha256": sorted({record["sha256"] for record in by_group[group]}),
                **_summary(by_group[group]),
            }
            for group in sorted(group_split)
        ],
        "duplicate_sha256": [
            {
                "sha256": sha,
                "ids": [r["id"] for r in values],
                "video_groups": sorted({r["video_group"] for r in values}),
                "split": values[0]["split"],
            }
            for sha, values in sorted(by_sha.items())
            if len(values) > 1
        ],
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    """Run dataset preparation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="source dataset directory")
    parser.add_argument(
        "--output", type=Path, help="output directory (defaults to dataset/prepared/uav_normal_640x352_seed0)"
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else source.parents[2] / "prepared" / OUTPUT_NAME
    stats = prepare_dataset(source, output)
    print(f"Prepared {stats['overall']['images']} images in {output}")


if __name__ == "__main__":
    main()

"""Validate COCO boxes, split source groups, and export a YOLO dataset.

Splitting happens before tiling. Duplicate decoded RGB images are kept once,
and groups connected by duplicate images are assigned to the same split. Every
positive box/tile intersection is retained, including small edge fragments;
therefore a tile containing a labeled object is never exported as background.
Only images explicitly listed in COCO are eligible as negative examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml
from PIL import Image


SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.7, 0.2, 0.1)


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class SourceImage:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    group_id: str
    sha256: str
    boxes: list[Box]
    reencode: bool = False


def require_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer: {value!r}")
    return value


def safe_image_path(images: Path, file_name: Any) -> Path:
    """Reject absolute paths, traversal, and symlinks escaping the image root."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Every COCO image needs a nonempty file_name")
    relative = Path(file_name.replace("\\", "/"))
    windows = PureWindowsPath(file_name)
    if relative.is_absolute() or windows.drive or windows.root or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {file_name!r}")
    root = images.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Image path escapes images directory: {file_name!r}")
    if not candidate.is_file():
        raise ValueError(f"Image file does not exist: {candidate}")
    return candidate


def clip_coco_box(bbox: Any, width: int, height: int, class_id: int) -> Box:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be [x, y, width, height]: {bbox!r}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        raise ValueError(f"bbox coordinates must be numbers: {bbox!r}")
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"bbox must be finite and have positive area: {bbox!r}")
    right, bottom = x + w, y + h
    if not math.isfinite(right) or not math.isfinite(bottom):
        raise ValueError(f"bbox overflows image coordinates: {bbox!r}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), right), min(float(height), bottom)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox has no positive intersection with the image: {bbox!r}")
    return Box(class_id, x1, y1, x2, y2)


def yolo_line(box: Box, width: int, height: int) -> str:
    values = ((box.x1 + box.x2) / (2 * width),
              (box.y1 + box.y2) / (2 * height),
              (box.x2 - box.x1) / width, (box.y2 - box.y1) / height)
    return f"{box.class_id} " + " ".join(f"{v:.12g}" for v in values)


def read_groups(path: Path | None, image_ids: set[int]) -> dict[int, str]:
    if path is None:
        return {image_id: f"image:{image_id}" for image_id in image_ids}
    result: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_id", "group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("groups.csv needs image_id,group_id columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                image_id = int(row["image_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid image_id in groups.csv row {row_number}") from exc
            group_id = (row["group_id"] or "").strip()
            if image_id not in image_ids or image_id in result or not group_id:
                raise ValueError(f"Unknown/duplicate image_id or empty group_id in groups.csv row {row_number}")
            result[image_id] = group_id
    missing = image_ids - result.keys()
    if missing:
        raise ValueError(f"groups.csv does not cover all COCO images; missing IDs: {sorted(missing)[:20]}")
    return result


def read_sources(annotations: Path, images: Path, groups: Path | None,
                 exif_policy: str = "reject", first_frame: bool = False) -> tuple[list[SourceImage], list[dict], dict]:
    document = json.loads(annotations.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("COCO root must be an object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"COCO {key!r} must be a list")
    categories_by_id = {}
    for category in document["categories"]:
        source_id = require_id(category.get("id"), "category.id")
        name = category.get("name")
        if source_id in categories_by_id or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Duplicate category ID or missing category name: {category!r}")
        categories_by_id[source_id] = name.strip()
    if not categories_by_id or len(set(categories_by_id.values())) != len(categories_by_id):
        raise ValueError("Category names must be nonempty and unique")
    categories = [{"source_id": source_id, "id": new_id, "name": categories_by_id[source_id]}
                  for new_id, source_id in enumerate(sorted(categories_by_id))]
    mapping = {category["source_id"]: category["id"] for category in categories}
    metadata = {}
    for entry in document["images"]:
        image_id = require_id(entry.get("id"), "image.id")
        if image_id in metadata:
            raise ValueError(f"Duplicate image ID: {image_id}")
        width = require_id(entry.get("width"), f"image {image_id} width")
        height = require_id(entry.get("height"), f"image {image_id} height")
        if not width or not height:
            raise ValueError(f"Image {image_id} has zero dimensions")
        metadata[image_id] = entry
    if not metadata:
        raise ValueError("COCO contains no images")
    image_groups = read_groups(groups, set(metadata))
    boxes: dict[int, list[Box]] = defaultdict(list)
    annotation_ids = set()
    clipped_count = 0
    crowd_count = 0
    for annotation in document["annotations"]:
        annotation_id = require_id(annotation.get("id"), "annotation.id")
        image_id = require_id(annotation.get("image_id"), "annotation.image_id")
        category_id = require_id(annotation.get("category_id"), "annotation.category_id")
        if annotation_id in annotation_ids or image_id not in metadata or category_id not in mapping:
            raise ValueError(f"Duplicate annotation ID or invalid image/category reference: annotation {annotation_id}")
        annotation_ids.add(annotation_id)
        entry = metadata[image_id]
        try:
            box = clip_coco_box(annotation.get("bbox"), entry["width"], entry["height"], mapping[category_id])
        except ValueError as exc:
            raise ValueError(f"Invalid annotation {annotation_id}: {exc}") from exc
        x, y, w, h = annotation["bbox"]
        clipped_count += (box.x1 != x or box.y1 != y or box.x2 != x + w or box.y2 != y + h)
        crowd_count += bool(annotation.get("iscrowd", 0))
        boxes[image_id].append(box)
    sources = []
    exif_count = frame_count = jpeg_repair_count = 0
    for image_id, entry in sorted(metadata.items()):
        path = safe_image_path(images, entry.get("file_name"))
        try:
            with Image.open(path) as image:
                has_orientation = image.getexif().get(274, 1) != 1
                if has_orientation and image.format == "TIFF":
                    raise ValueError(f"Image {image_id} has oriented TIFF pixels; Pillow may rotate them on decode. Normalize TIFF and annotations explicitly first.")
                if has_orientation and exif_policy == "reject":
                    raise ValueError(f"Image {image_id} has EXIF orientation; normalize image AND boxes before export")
                if image.size != (entry["width"], entry["height"]):
                    raise ValueError(f"Image {image_id} dimensions disagree: COCO {(entry['width'], entry['height'])}, actual {image.size}")
                has_frames = getattr(image, "n_frames", 1) != 1
                if has_frames and not first_frame:
                    raise ValueError(f"Image {image_id} has multiple frames; export one frame with matching annotations")
                image.seek(0)
                pixels = image.convert("RGB")
                digest = hashlib.sha256(struct.pack(">II", *pixels.size) + pixels.tobytes()).hexdigest()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Cannot decode image {image_id}: {path}") from exc
        needs_jpeg_repair = False
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            with path.open("rb") as handle:
                handle.seek(-2, 2)
                needs_jpeg_repair = handle.read() != b"\xff\xd9"
        exif_count += has_orientation
        frame_count += has_frames
        jpeg_repair_count += needs_jpeg_repair
        sources.append(SourceImage(image_id, entry["file_name"], path, entry["width"], entry["height"],
                                   image_groups[image_id], digest, boxes[image_id],
                                   has_orientation or has_frames or needs_jpeg_repair))
    validation = {"annotations": len(annotation_ids), "clipped_boxes": clipped_count, "crowd_boxes": crowd_count,
                  "exif_images": exif_count, "multi_frame_images": frame_count,
                  "jpeg_end_marker_repairs": jpeg_repair_count}
    return sources, categories, validation


def deduplicate_and_split(sources: list[SourceImage], seed: int) -> tuple[list[SourceImage], dict[int, str], dict[int, int], int]:
    """Keep complete groups together, including groups connected by duplicates."""
    parents = {source.group_id: source.group_id for source in sources}

    def find(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    by_hash: dict[str, SourceImage] = {}
    duplicate_of: dict[int, int] = {}
    representatives = []
    for source in sorted(sources, key=lambda value: value.image_id):
        if source.sha256 not in by_hash:
            by_hash[source.sha256] = source
            representatives.append(source)
            continue
        original = by_hash[source.sha256]
        box_key = lambda box: (box.class_id, box.x1, box.y1, box.x2, box.y2)
        if sorted(map(box_key, source.boxes)) != sorted(map(box_key, original.boxes)):
            raise ValueError(f"Duplicate pixels have conflicting annotations: image {original.image_id} and {source.image_id}")
        first, second = sorted((find(original.group_id), find(source.group_id)))
        parents[second] = first
        duplicate_of[source.image_id] = original.image_id
    components: dict[str, list[SourceImage]] = defaultdict(list)
    for source in representatives:
        components[find(source.group_id)].append(source)
    group_keys = sorted(components)
    if len(group_keys) < 3:
        raise ValueError("At least 3 independent groups remain after duplicate removal; need 3 or more for train/val/test. "
                         f"Found {len(group_keys)}. Add data or revise verified scene groups.")
    random.Random(seed).shuffle(group_keys)
    cumulative = [0]
    for key in group_keys:
        cumulative.append(cumulative[-1] + len(components[key]))
    total = len(representatives)
    train_end = min(range(1, len(group_keys) - 1), key=lambda index: abs(cumulative[index] - total * 0.7))
    val_end = min(range(train_end + 1, len(group_keys)), key=lambda index: abs(cumulative[index] - total * 0.9))
    component_split = {}
    for index, key in enumerate(group_keys):
        component_split[key] = "train" if index < train_end else "val" if index < val_end else "test"
    assignments = {source.image_id: component_split[find(source.group_id)] for source in sources}
    return representatives, assignments, duplicate_of, len(components)


def tile_windows(width: int, height: int, tile_size: int | None, overlap: float) -> list[tuple[int, int, int, int]]:
    if tile_size is None:
        return [(0, 0, width, height)]
    stride = max(1, round(tile_size * (1 - overlap)))

    def starts(length: int) -> list[int]:
        end = max(0, length - tile_size)
        positions = list(range(0, end + 1, stride))
        if positions[-1] != end:
            positions.append(end)
        return positions

    return [(x, y, min(width, x + tile_size), min(height, y + tile_size))
            for y in starts(height) for x in starts(width)]


def intersect_boxes(boxes: list[Box], window: tuple[int, int, int, int]) -> list[Box]:
    left, top, right, bottom = window
    result = []
    for box in boxes:
        x1, y1 = max(box.x1, left), max(box.y1, top)
        x2, y2 = min(box.x2, right), min(box.y2, bottom)
        if x2 > x1 and y2 > y1:
            result.append(Box(box.class_id, x1 - left, y1 - top, x2 - left, y2 - top))
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(annotations: Path, images: Path, output: Path, groups: Path | None = None,
                    *, allow_image_split: bool = False, tile_size: int | None = None,
                    overlap: float = 0.2, seed: int = 42, exif_policy: str = "reject",
                    first_frame: bool = False) -> dict:
    annotations, images, output = Path(annotations), Path(images), Path(output)
    groups = Path(groups) if groups is not None else None
    if exif_policy not in {"reject", "raw-pixels"}:
        raise ValueError("exif_policy must be reject or raw-pixels")
    if groups is None and not allow_image_split:
        raise ValueError("Provide verified --groups CSV (image_id,group_id), or explicitly use --allow-image-split "
                         "for an exploratory run. Image-level splitting does NOT prevent flight/scene leakage.")
    if tile_size is not None and (isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0):
        raise ValueError("tile_size must be a positive integer")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and in [0, 1)")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory; refusing to mix datasets: {output}")
    if output.resolve().is_relative_to(images.resolve()):
        raise ValueError("Output cannot be inside the source images directory")
    sources, categories, validation = read_sources(annotations, images, groups, exif_policy, first_frame)
    representatives, assignments, duplicates, component_count = deduplicate_and_split(sources, seed)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    by_split = {split: {"images": 0, "objects": 0, "negative_images": 0, "source_images": 0} for split in SPLITS}
    tile_rows = []
    for source in representatives:
        split = assignments[source.image_id]
        by_split[split]["source_images"] += 1
        windows = tile_windows(source.width, source.height, tile_size, overlap)
        pixels = None
        if tile_size is not None or source.reencode:
            with Image.open(source.path) as source_file:
                pixels = source_file.convert("RGB")
        for tile_index, window in enumerate(windows):
            left, top, right, bottom = window
            width, height = right - left, bottom - top
            boxes = intersect_boxes(source.boxes, window)
            stem = f"image_{source.image_id:08d}"
            if tile_size is not None:
                stem += f"_tile_{tile_index:05d}"
            extension = ".png" if tile_size is not None or source.reencode else source.path.suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                extension = ".png"
            image_name = stem + extension
            image_target = output / "images" / split / image_name
            if pixels is not None:
                tile = pixels.crop(window)
                tile.info.clear()  # PNG can retain EXIF too: remove it explicitly.
                tile.save(image_target)
            elif extension == source.path.suffix.lower():
                shutil.copyfile(source.path, image_target)
            else:
                with Image.open(source.path) as source_file:
                    source_file.convert("RGB").save(image_target)
            lines = [yolo_line(box, width, height) for box in boxes]
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            by_split[split]["images"] += 1
            by_split[split]["objects"] += len(boxes)
            by_split[split]["negative_images"] += not boxes
            tile_rows.append({"image_id": source.image_id, "source_file": source.file_name, "split": split,
                              "output_file": f"images/{split}/{image_name}", "x": left, "y": top,
                              "width": width, "height": height, "objects": len(boxes)})
        if pixels is not None:
            pixels.close()
    split_rows = [{"image_id": source.image_id, "file_name": source.file_name, "group_id": source.group_id,
                   "split": assignments[source.image_id], "sha256": source.sha256,
                   "duplicate_of": duplicates.get(source.image_id, "")}
                  for source in sources]
    write_csv(output / "split_manifest.csv", split_rows,
              ["image_id", "file_name", "group_id", "split", "sha256", "duplicate_of"])
    write_csv(output / "tiles.csv", tile_rows,
              ["image_id", "source_file", "split", "output_file", "x", "y", "width", "height", "objects"])
    # Without path, Ultralytics resolves the root from this YAML's parent directory.
    data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test",
                 "names": {category["id"]: category["name"] for category in categories}}
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    limitations = ["COCO-listed images without annotations are treated as negatives; annotation completeness must be verified.",
                  "Exact RGB duplicates are removed; near-duplicates require verified flight/scene groups.",
                  "Actual split ratios can differ from 70/20/10 because complete groups are kept together."]
    if groups is None:
        limitations.insert(0, "EXPLORATORY IMAGE SPLIT: flight/scene leakage is NOT prevented. Do not report these test scores as independent generalization.")
    else:
        limitations.append("Group isolation is only as reliable as the supplied group metadata; verify flight/location boundaries.")
        with groups.open(encoding="utf-8-sig", newline="") as handle:
            bases = sorted({row.get("basis", "unspecified") for row in csv.DictReader(handle)})
        if any("unverified" in basis or "heuristic" in basis for basis in bases):
            limitations.append("UNVERIFIED FILENAME GROUPS: series grouping is not confirmed flight/location separation; scores are preliminary.")
    if exif_policy == "raw-pixels":
        limitations.append("EXIF orientation is removed without rotating pixels; annotations must refer to raw pixel coordinates.")
    if first_frame:
        limitations.append("Only frame 0 of multi-frame images is exported; annotations must describe that frame.")
    if tile_size is not None:
        limitations.append("All positive box/tile intersections are retained, including small edge fragments. Tiling increases box counts and does not improve sensor resolution.")
    if validation["crowd_boxes"]:
        limitations.append("COCO crowd boxes are exported as ordinary boxes; YOLO labels cannot preserve iscrowd semantics.")
    source_class_counts = Counter(box.class_id for source in representatives for box in source.boxes)
    manifest = {
        "schema_version": 1,
        "source_annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "groups_sha256": hashlib.sha256(groups.read_bytes()).hexdigest() if groups else None,
        "source_annotations": str(annotations.resolve()), "source_images": str(images.resolve()),
        "seed": seed, "split_strategy": "group" if groups else "exploratory_image",
        "split_ratios": dict(zip(SPLITS, DEFAULT_RATIOS)), "independent_groups": component_count,
        "categories": categories,
        "category_mapping": {str(category["source_id"]): category["id"] for category in categories},
        "counts": {"source_images": len(sources), "unique_images": len(representatives),
                   "duplicates_removed": len(duplicates), "output_images": sum(row["images"] for row in by_split.values()),
                   "objects": sum(row["objects"] for row in by_split.values()),
                   "unique_source_objects": sum(source_class_counts.values()),
                   "source_objects_by_class": {str(category["id"]): source_class_counts[category["id"]] for category in categories},
                   "by_split": by_split},
        "validation": validation,
        "image_policy": {"exif": exif_policy, "first_frame": first_frame, "repair_jpeg_end_marker": True},
        "pixel_hash": "sha256(big-endian uint32 width,height + decoded RGB bytes)",
        "tiling": {"enabled": tile_size is not None, "tile_size": tile_size, "overlap": overlap if tile_size else None,
                   "box_policy": "retain_every_positive_intersection"},
        "limitations": limitations,
        "files": {"data_yaml": "data.yaml", "split_manifest": "split_manifest.csv", "tiles": "tiles.csv"},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=Path, help="CSV with image_id,group_id; use flight/scene/location groups")
    parser.add_argument("--allow-image-split", action="store_true", help="Explicit exploratory fallback; scene leakage may remain")
    parser.add_argument("--tile-size", type=int, help="Optional square tile size in source pixels")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exif-policy", choices=("reject", "raw-pixels"), default="reject",
                        help="raw-pixels strips EXIF without rotating pixels; use only with matching COCO coordinates")
    parser.add_argument("--first-frame", action="store_true",
                        help="Explicitly select frame 0 from multi-frame images with matching annotations")
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    for limitation in manifest["limitations"]:
        print(f"NOTE: {limitation}")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

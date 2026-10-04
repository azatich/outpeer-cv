"""Create reproducible EDA tables, plots and annotated previews for a COCO dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

from src.data.download import safe_path, sequence_group


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def audit(annotations: Path, images: Path, output: Path, samples: int = 24, seed: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(annotations.read_text(encoding="utf-8"))
    records = {record["id"]: record for record in data["images"]}
    categories = {record["id"]: record["name"] for record in data["categories"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    object_counts: Counter = Counter()
    image_sets: dict[int, set] = defaultdict(set)
    areas, sides_at_640, box_rows, issues = [], [], [], []
    for annotation in data["annotations"]:
        image_id, category_id = annotation["image_id"], annotation["category_id"]
        if image_id not in records or category_id not in categories:
            issues.append({"annotation_id": annotation.get("id"), "error": "unknown image or category"})
            continue
        record = records[image_id]
        bbox = annotation.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)):
            issues.append({"annotation_id": annotation.get("id"), "error": "invalid numeric bbox"})
            continue
        x, y, width, height = bbox
        if width <= 0 or height <= 0 or record["width"] <= 0 or record["height"] <= 0:
            issues.append({"annotation_id": annotation.get("id"), "error": "nonpositive dimensions"})
            continue
        if x < 0 or y < 0 or x + width > record["width"] + 1 or y + height > record["height"] + 1:
            issues.append({"annotation_id": annotation.get("id"), "error": "bbox exceeds image"})
        relative_area = width * height / (record["width"] * record["height"])
        min_side = min(width, height) * 640 / max(record["width"], record["height"])
        areas.append(relative_area)
        sides_at_640.append(min_side)
        object_counts[category_id] += 1
        image_sets[category_id].add(image_id)
        by_image[image_id].append(annotation)
        box_rows.append({"image_id": image_id, "category": categories[category_id],
                         "width_px": width, "height_px": height, "relative_area": relative_area,
                         "min_side_after_resize_640": min_side})
    image_rows = []
    readable = []
    for image_id, record in records.items():
        path = safe_path(images, record["file_name"])
        status = "ok"
        orientation, frames = None, None
        try:
            with Image.open(path) as image:
                actual = image.size
                orientation = image.getexif().get(274, 1)
                oriented_tiff = image.format == "TIFF" and orientation != 1
                frames = getattr(image, "n_frames", 1)
                image.load()
            if actual != (record["width"], record["height"]):
                status = "dimension_mismatch"
            elif oriented_tiff:
                status = "oriented_tiff_requires_normalization"
            elif orientation != 1:
                status = "exif_orientation_requires_review"
            elif frames != 1:
                status = "multiple_frames_requires_review"
        except (OSError, ValueError):
            status = "missing_or_corrupt"
        if status not in {"missing_or_corrupt", "dimension_mismatch", "oriented_tiff_requires_normalization"}:
            readable.append(record)
        image_rows.append({"image_id": image_id, "file_name": record["file_name"],
                           "width": record["width"], "height": record["height"],
                           "objects": len(by_image[image_id]), "status": status,
                           "exif_orientation": orientation, "frames": frames,
                           "suggested_group": sequence_group(record["file_name"])})
    class_rows = [{"class_id": key, "name": name, "objects": object_counts[key],
                   "images": len(image_sets[key])} for key, name in categories.items()]
    write_csv(output / "classes.csv", class_rows, ["class_id", "name", "objects", "images"])
    write_csv(output / "images.csv", image_rows,
              ["image_id", "file_name", "width", "height", "objects", "status", "exif_orientation", "frames", "suggested_group"])
    write_csv(output / "boxes.csv", box_rows,
              ["image_id", "category", "width_px", "height_px", "relative_area", "min_side_after_resize_640"])
    summary = {
        "images": len(records), "annotations": len(data["annotations"]),
        "readable_images": len(readable), "classes": class_rows,
        "images_without_annotations": sum(not by_image[key] for key in records),
        "image_statuses": dict(Counter(row["status"] for row in image_rows)),
        "annotation_issues": issues,
        "objects_min_side_below_8px_at_640": sum(side < 8 for side in sides_at_640),
        "small_object_definition": "Shorter box side <8px after aspect-preserving resize to 640px longest side; diagnostic, not COCO small-object AP.",
        "filename_groups": dict(Counter(row["suggested_group"] for row in image_rows)),
        "limitations": ["Filename groups are not verified flights or geographic locations.",
                        "UAVVaste is a one-class litter dataset; no material classification or satellite capability is established.",
                        "Marine/coastal generalization requires a separate representative labelled dataset."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar([r["name"] for r in class_rows], [r["objects"] for r in class_rows])
    axes[0, 0].set(title="Objects by class", ylabel="Objects")
    axes[0, 1].hist([row["objects"] for row in image_rows], bins=25)
    axes[0, 1].set(title="Objects per image", xlabel="Objects", ylabel="Images")
    axes[1, 0].hist(sides_at_640, bins=40)
    axes[1, 0].axvline(8, color="red", linestyle="--", label="8 px")
    axes[1, 0].set(title="Shorter box side after resize to 640", xlabel="Pixels", ylabel="Objects")
    axes[1, 0].legend()
    axes[1, 1].scatter([r["width"] for r in image_rows], [r["height"] for r in image_rows], alpha=0.3)
    axes[1, 1].set(title="Original image dimensions", xlabel="Width (px)", ylabel="Height (px)")
    fig.tight_layout()
    fig.savefig(output / "distributions.png", dpi=140)
    plt.close(fig)

    def thumbnail(record: dict, width: int = 480, height: int = 290) -> Image.Image:
        with Image.open(safe_path(images, record["file_name"])) as source:
            image = source.convert("RGB")
        # Draw on the original geometry; do not apply EXIF rotations independently of boxes.
        draw = ImageDraw.Draw(image)
        for annotation in by_image[record["id"]]:
            x, y, w, h = annotation["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="#ff3030", width=max(3, image.width // 500))
        preview = Image.new("RGB", (width, height + 34), "white")
        image = ImageOps.contain(image, (width, height))
        preview.paste(image, ((width - image.width) // 2, 0))
        ImageDraw.Draw(preview).text((6, height + 5), record["file_name"], fill="black")
        return preview

    if readable:
        selected = random.Random(seed).sample(readable, min(samples, len(readable)))
        grid = Image.new("RGB", (480 * 4, 324 * ((len(selected) + 3) // 4)), "white")
        for index, record in enumerate(selected):
            grid.paste(thumbnail(record), ((index % 4) * 480, (index // 4) * 324))
        grid.save(output / "annotated_examples.jpg", quality=88)
        grouped = defaultdict(list)
        for record in readable:
            grouped[sequence_group(record["file_name"])].append(record)
        sheet_dir = output / "contact_sheets"
        sheet_dir.mkdir(exist_ok=True)
        for name, group in sorted(grouped.items()):
            representatives = group[::max(1, len(group) // 8)][:8]
            sheet = Image.new("RGB", (480 * 4, 324 * ((len(representatives) + 3) // 4)), "white")
            for index, record in enumerate(representatives):
                sheet.paste(thumbnail(record), ((index % 4) * 480, (index // 4) * 324))
            sheet.save(sheet_dir / f"{name}.jpg", quality=85)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/uavvaste/annotations.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/uavvaste/images"))
    parser.add_argument("--output", type=Path, default=Path("reports/a/audit"))
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    summary = audit(args.annotations, args.images, args.output, args.samples)
    print(json.dumps({key: summary[key] for key in ("images", "annotations", "readable_images", "image_statuses")}, indent=2))


if __name__ == "__main__":
    main()

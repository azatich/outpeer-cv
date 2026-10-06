"""Download official TACO labels/images and prepare bottle, bag and can detection.

Only human-reviewed annotations.json is used. Target categories retain their
original boxes; other TACO objects are outside this three-class task. Images
without target objects are included as backgrounds, not called clean scenes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import shutil
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from src.data.prepare import prepare_dataset

REVISION = "29de1a9ba05a647b83a90f18d7772e20bb23d846"
ANNOTATIONS_URL = f"https://raw.githubusercontent.com/pedropro/TACO/{REVISION}/data/annotations.json"
ANNOTATIONS_SHA256 = "ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449"
CLASS_MAPPING = {
    "bottle": ("Other plastic bottle", "Clear plastic bottle", "Glass bottle"),
    "bag": ("Paper bag", "Plastified paper bag", "Garbage bag", "Single-use carrier bag", "Polypropylene bag"),
    "can": ("Food Can", "Aerosol", "Drink can"),
}


def subset(document: dict, background_images: int = 150, seed: int = 42) -> dict:
    """Map genuine source labels; include deterministic background examples."""
    if background_images < 0:
        raise ValueError("background_images must be nonnegative")
    source_ids = {c["name"]: c["id"] for c in document["categories"]}
    mapping = {}
    for target, names in enumerate(CLASS_MAPPING.values()):
        for name in names:
            mapping[source_ids[name]] = target
    annotations = [{**a, "category_id": mapping[a["category_id"]]}
                   for a in document["annotations"] if a["category_id"] in mapping]
    positive_ids = {a["image_id"] for a in annotations}
    # Spread background selection across source batches rather than selecting
    # only the first batch. None of their non-target litter becomes a target.
    backgrounds = defaultdict(list)
    for image in document["images"]:
        if image["id"] not in positive_ids:
            backgrounds[Path(image["file_name"]).parts[0]].append(image)
    rng = random.Random(seed)
    for images in backgrounds.values():
        rng.shuffle(images)
    background_ids = []
    while len(background_ids) < background_images and any(backgrounds.values()):
        for group in sorted(backgrounds):
            if backgrounds[group] and len(background_ids) < background_images:
                background_ids.append(backgrounds[group].pop()["id"])
    selected = positive_ids | set(background_ids)
    return {"info": {"source": "TACO", "source_revision": REVISION,
                     "class_mapping": CLASS_MAPPING, "background_images_requested": background_images},
            "licenses": document.get("licenses", []),
            "images": [im for im in document["images"] if im["id"] in selected],
            "annotations": annotations,
            "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_MAPPING)]}


def download_image(image: dict, root: Path) -> dict:
    relative = Path(image["file_name"])
    if relative.is_absolute() or ".." in relative.parts or relative.drive:
        raise ValueError("Unsafe TACO file name")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes TACO root")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (image["width"], image["height"])

    def check_dimensions(pixels: Image.Image) -> None:
        if pixels.size == expected:
            return
        # Flickr's 640px derivative has the same field of view. Allow one-pixel
        # aspect-ratio rounding, never an unknown crop or a rotated photograph.
        w, h = pixels.size
        if abs(w * expected[1] - h * expected[0]) > 2 * max(expected):
            raise ValueError(f"Source aspect ratio differs for {relative}: {pixels.size} != {expected}")

    if target.is_file():
        with Image.open(target) as pixels:
            pixels.load()
            oriented = ImageOps.exif_transpose(pixels)
            check_dimensions(oriented)
            size = oriented.size
        return {"image_id": image["id"], "file": relative.as_posix(), "cached": True,
                "size": size, "source_size": expected,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    error = None
    urls = [image.get("flickr_640_url"), image["flickr_url"]]
    for attempt in range(2):
        for original_url in urls:
            if not original_url:
                continue
            url = original_url.replace("http://", "https://", 1)
            try:
                request = Request(url, headers={"User-Agent": "outpeer-cv/1.0"})
                with urlopen(request, timeout=20) as response:
                    payload = response.read()
                with Image.open(BytesIO(payload)) as pixels:
                    pixels.load()
                    oriented = ImageOps.exif_transpose(pixels)
                    check_dimensions(oriented)
                    size = oriented.size
                partial = target.with_suffix(target.suffix + ".part")
                partial.write_bytes(payload)
                partial.replace(target)
                return {"image_id": image["id"], "file": relative.as_posix(), "url": url,
                        "size": size, "source_size": expected,
                        "sha256": hashlib.sha256(payload).hexdigest()}
            except Exception as exc:
                error = exc
        time.sleep(attempt + 1)
    raise RuntimeError(f"Cannot download {relative}: {error}")


def rescale_annotations(document: dict, sizes: dict[int, tuple[int, int]]) -> None:
    """Scale real boxes/polygons to the corresponding normalized image pixels."""
    scales = {}
    for image in document["images"]:
        width, height = sizes[image["id"]]
        scales[image["id"]] = (width / image["width"], height / image["height"])
        image["original_width"], image["original_height"] = image["width"], image["height"]
        image["width"], image["height"] = width, height
    for annotation in document["annotations"]:
        sx, sy = scales[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        annotation["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        if "area" in annotation:
            annotation["area"] *= sx * sy
        if isinstance(annotation.get("segmentation"), list):
            annotation["segmentation"] = [[v * (sx if i % 2 == 0 else sy)
                                            for i, v in enumerate(polygon)] for polygon in annotation["segmentation"]]


def download(raw: Path, *, workers: int = 8, background_images: int = 150) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    raw.mkdir(parents=True, exist_ok=True)
    original = raw / "annotations_original.json"
    if not original.is_file():
        with urlopen(ANNOTATIONS_URL, timeout=60) as response:
            original.write_bytes(response.read())
    if hashlib.sha256(original.read_bytes()).hexdigest() != ANNOTATIONS_SHA256:
        raise ValueError("Original annotations differ from the pinned official TACO revision")
    original_document = json.loads(original.read_text(encoding="utf-8-sig"))
    chosen = subset(original_document, background_images)
    rows, failures = [], []
    images_root = raw / "images"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_image, image, images_root): image for image in chosen["images"]}
        for index, future in enumerate(as_completed(futures), 1):
            image = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"image_id": image["id"], "error": str(exc)})
            if index % 25 == 0 or index == len(futures):
                print(f"TACO images: {index}/{len(futures)}; unavailable: {len(failures)}", flush=True)
    available = {row["image_id"] for row in rows}
    # Expired source URLs are explicitly recorded and their annotations removed.
    # Exclusions are made before splitting; no validation/test metrics guide them.
    chosen["images"] = [im for im in chosen["images"] if im["id"] in available]
    chosen["annotations"] = [a for a in chosen["annotations"] if a["image_id"] in available]
    sizes = {}
    for image in chosen["images"]:
        relative = Path(image["file_name"])
        normalized = relative.with_suffix(".png")
        destination = raw / "normalized" / normalized
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(images_root / relative) as source:
            pixels = ImageOps.exif_transpose(source).convert("RGB")
            pixels.thumbnail((640, 640), Image.Resampling.LANCZOS)
            pixels.info.clear()
            pixels.save(destination)
            sizes[image["id"]] = pixels.size
        image["original_file_name"] = image["file_name"]
        image["file_name"] = normalized.as_posix()
    rescale_annotations(chosen, sizes)
    (raw / "annotations.json").write_text(json.dumps(chosen, ensure_ascii=False), encoding="utf-8")
    with (raw / "groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "group_id", "basis"])
        writer.writeheader()
        writer.writerows({"image_id": im["id"], "group_id": Path(im["file_name"]).parts[0],
                          "basis": "unverified_source_batch"} for im in chosen["images"])
    manifest = {"source": "https://github.com/pedropro/TACO", "source_revision": REVISION,
                "annotations_url": ANNOTATIONS_URL, "annotations_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "selected_annotations_sha256": hashlib.sha256((raw / "annotations.json").read_bytes()).hexdigest(),
                "class_mapping": CLASS_MAPPING, "images": len(chosen["images"]),
                "objects": len(chosen["annotations"]), "objects_by_class": dict(Counter(a["category_id"] for a in chosen["annotations"])),
                "downloads": sorted(rows, key=lambda row: row["image_id"]), "unavailable": failures}
    (raw / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(chosen["images"]) < 100:
        raise RuntimeError("Too few source images available for a useful experiment")
    return manifest


def prepare(raw: Path, output: Path, reference: Path) -> dict:
    manifest = prepare_dataset(raw / "annotations.json", raw / "normalized", output,
                               groups=raw / "groups.csv", exif_policy="reject", first_frame=True)
    manifest["source"] = json.loads((raw / "download_manifest.json").read_text())
    manifest["limitations"] += [
        "Only bottle, bag and can are target objects; other litter is not classified by this model.",
        "Background images can contain non-target litter; they are not necessarily clean.",
        "Source batches are separated, but independent locations/photographers are not verified.",
        "Available Flickr 640px derivatives are used with explicit coordinate scaling; originals are a fallback.",
        "EXIF display orientation is applied as in the official TACO loader; normalized RGB PNGs have longest side <=640px.",
        "TACO includes ground-level outdoor images; this experiment does not validate drone, marine or satellite generalization.",
    ]
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Freeze a reference without large image bytes; evaluation checks split and pixels.
    reference.mkdir(parents=True, exist_ok=False)
    for name in ("dataset_manifest.json", "split_manifest.csv", "tiles.csv", "data.yaml"):
        shutil.copy2(output / name, reference / name)
    counts = {split: Counter() for split in ("train", "val", "test")}
    for split in counts:
        for label in (output / "labels" / split).glob("*.txt"):
            counts[split].update(int(line.split()[0]) for line in label.read_text().splitlines())
        if set(counts[split]) != {0, 1, 2}:
            raise ValueError(f"All classes must be present in {split}: {counts[split]}")
    summary = {"images": manifest["counts"], "objects_by_split_and_class": counts,
               "classes": list(CLASS_MAPPING), "class_mapping": CLASS_MAPPING,
               "source_revision": REVISION, "unavailable_images": len(manifest["source"]["unavailable"])}
    (reference / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/taco"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/taco_types"))
    parser.add_argument("--reference", type=Path, default=Path("reports/multiclass/dataset"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--background-images", type=int, default=150)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    download(args.raw, workers=args.workers, background_images=args.background_images)
    if not args.download_only:
        print(json.dumps(prepare(args.raw, args.output, args.reference), indent=2))


if __name__ == "__main__":
    main()

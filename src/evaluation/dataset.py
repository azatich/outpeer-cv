"""Check prepared data against a frozen split/pixel reference before experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

from PIL import Image
import yaml

from src.inference.predict import ROOT, local_path, validate_classes
from src.training.train import sha256, _class_names


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: json.dumps(row, sort_keys=True))


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Truth:
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass
class Sample:
    image: Path
    relative: str
    width: int
    height: int
    truth: list[Truth]


def read_labels(path: Path, width: int, height: int, num_classes: int = 1) -> list[Truth]:
    """Missing labels are an error, not silently treated as a negative scene."""
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label: {path}: {line}")
        cls, x, y, w, h = map(float, parts)
        if not all(math.isfinite(v) for v in (cls, x, y, w, h)) or cls != int(cls) or not 0 <= cls < num_classes:
            raise ValueError(f"Expected finite coordinates and a class in [0,{num_classes-1}]: {path}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Out-of-range normalized box: {path}")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"Box extends beyond image: {path}")
        result.append(Truth(int(cls), (max(0, (x-w/2)*width), max(0, (y-h/2)*height),
                                min(width, (x+w/2)*width), min(height, (y+h/2)*height))))
    return result


class PreparedDataset:
    def __init__(self, data: str | Path, reference: str | Path = "reports/a/dataset"):
        self.path = local_path(data)
        self.root = self.path.parent
        self.reference = local_path(reference)
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8-sig"))
        root = Path(raw.get("path", "."))
        root = (root if root.is_absolute() else self.root / root).resolve()
        self.names = dict(enumerate(_class_names(raw.get("names"))))
        validate_classes(self.names)
        if root != self.root:
            raise ValueError("Expected prepared dataset root")
        for split in ("train", "val", "test"):
            if not isinstance(raw.get(split), str) or (root / raw[split]).resolve() != root / "images" / split:
                raise ValueError(f"Expected images/{split} in data.yaml")
        self.manifest = read_json(root / "dataset_manifest.json")
        if {c["id"]: c["name"] for c in self.manifest["categories"]} != self.names:
            raise ValueError("Dataset YAML class names differ from manifest")
        reference_manifest = read_json(self.reference / "dataset_manifest.json")
        for key in ("source_annotations_sha256", "seed", "categories", "image_policy", "tiling"):
            if self.manifest.get(key) != reference_manifest.get(key):
                raise ValueError(f"Dataset differs from frozen reference: {key}")
        if self.manifest.get("tiling", {}).get("enabled"):
            raise ValueError("This comparison requires the original untiled split")
        self.split_rows = csv_rows(root / "split_manifest.csv")
        self.tiles = csv_rows(root / "tiles.csv")
        if self.split_rows != csv_rows(self.reference / "split_manifest.csv"):
            raise ValueError("Split/pixel manifest differs from frozen reference; do not resplit the dataset")
        if self.tiles != csv_rows(self.reference / "tiles.csv"):
            raise ValueError("Prepared image mapping differs from frozen reference")
        labels = {}
        for row in self.tiles:
            relative = Path(row["output_file"])
            image = (root / relative).resolve()
            if not image.is_relative_to(root) or relative.parts[:2] != ("images", row["split"]):
                raise ValueError("Unsafe or inconsistent tile path")
            if not image.is_file():
                raise FileNotFoundError(image)
            label = root / "labels" / row["split"] / relative.with_suffix(".txt").name
            labels[str(label.relative_to(root)).replace("\\", "/")] = sha256(label)
        # Labels are hashed for identity only; test contents are interpreted only in load_split('test').
        self.identity = digest_json({"split": self.split_rows, "tiles": self.tiles, "labels": labels,
                                     "source": self.manifest["source_annotations_sha256"]})

    def resolved_yaml(self, split: str) -> dict:
        if split not in {"val", "test"}:
            raise ValueError("Evaluation split must be val or test")
        value = {"path": str(self.root), "names": self.names,
                 "train": str(self.root / "images/train"), "val": str(self.root / "images/val")}
        if split == "test":
            value["test"] = str(self.root / "images/test")
        return value

    def load_split(self, split: str) -> list[Sample]:
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        hashes = {row["image_id"]: row["sha256"] for row in self.split_rows}
        samples = []
        expected = sorted(row["output_file"] for row in self.tiles if row["split"] == split)
        actual = sorted(p.relative_to(self.root).as_posix() for p in (self.root / "images" / split).rglob("*") if p.is_file())
        if expected != actual:
            raise ValueError(f"Missing or extra images in {split}")
        for row in sorted(self.tiles, key=lambda row: row["output_file"]):
            if row["split"] != split:
                continue
            image_path = self.root / row["output_file"]
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                digest = hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
                if digest != hashes[row["image_id"]]:
                    raise ValueError(f"Prepared pixels differ from reference: {image_path}")
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f"Prepared images must have normalized EXIF: {image_path}")
                width, height = rgb.size
            label = self.root / "labels" / split / image_path.with_suffix(".txt").name
            truth = read_labels(label, width, height, len(self.names))
            if len(truth) != int(row["objects"]):
                raise ValueError(f"Object count differs from preparation: {label}")
            samples.append(Sample(image_path, row["output_file"], width, height, truth))
        if not samples:
            raise ValueError(f"Empty split: {split}")
        return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/uavvaste/data.yaml")
    parser.add_argument("--reference", default="reports/a/dataset")
    args = parser.parse_args()
    try:
        dataset = PreparedDataset(args.data, args.reference)
        print(json.dumps({"status": "verified", "identity": dataset.identity,
                          "counts": dataset.manifest["counts"]["by_split"]}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

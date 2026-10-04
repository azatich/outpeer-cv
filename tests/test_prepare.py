"""Small independent fixtures for preparation correctness and leakage barriers."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.prepare import Box, clip_coco_box, prepare_dataset, yolo_line


class PrepareTests(unittest.TestCase):
    def test_oriented_tiff_is_rejected_even_in_raw_pixel_mode(self):
        groups = self.fixture()
        image = Image.new("RGB", (12, 8), (0, 20, 100))
        image.putpixel((0, 0), (255, 0, 0))
        exif = Image.Exif()
        exif[274] = 2
        image.save(self.images / "1.tiff", exif=exif)
        document = json.loads(self.annotations.read_text())
        document["images"][0]["file_name"] = "1.tiff"
        self.annotations.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "oriented TIFF"):
            prepare_dataset(self.annotations, self.images, self.root / "oriented", groups, exif_policy="raw-pixels")

    def test_explicit_raw_exif_policy_preserves_pixels_and_box_coordinates(self):
        groups = self.fixture()
        path = self.images / "1.png"
        with Image.open(path) as source:
            image = source.copy()
        image.putpixel((0, 0), (255, 0, 0))
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        output = self.root / "raw-exif"
        manifest = prepare_dataset(self.annotations, self.images, output, groups, exif_policy="raw-pixels")
        tile = next(row for row in self.rows(output, "tiles.csv") if row["image_id"] == "1")
        with Image.open(output / tile["output_file"]) as exported:
            self.assertEqual(exported.size, (12, 8))
            self.assertEqual(exported.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(exported.getexif().get(274, 1), 1)
        self.assertEqual(manifest["validation"]["exif_images"], 1)

    def test_first_frame_requires_opt_in_and_exports_only_frame_zero(self):
        groups = self.fixture()
        first = Image.new("RGB", (12, 8), (0, 20, 100))
        first.save(self.images / "1.tiff", save_all=True,
                   append_images=[Image.new("RGB", (12, 8), "yellow")])
        document = json.loads(self.annotations.read_text())
        document["images"][0]["file_name"] = "1.tiff"
        self.annotations.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "multiple frames"):
            prepare_dataset(self.annotations, self.images, self.root / "reject-frames", groups)
        output = self.root / "first-frame"
        manifest = prepare_dataset(self.annotations, self.images, output, groups, first_frame=True)
        tile = next(row for row in self.rows(output, "tiles.csv") if row["image_id"] == "1")
        with Image.open(output / tile["output_file"]) as exported:
            self.assertEqual(exported.getpixel((0, 0)), (0, 20, 100))
            self.assertEqual(getattr(exported, "n_frames", 1), 1)
        self.assertEqual(manifest["validation"]["multi_frame_images"], 1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.annotations = self.root / "annotations.json"

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, count=6, duplicate=False, groups=True):
        images, annotations, group_rows = [], [], []
        for index in range(count):
            image_id = index + 1
            color = (index * 30, 20, 100)
            if duplicate and image_id == 2:
                color = (0, 20, 100)
            Image.new("RGB", (12, 8), color).save(self.images / f"{image_id}.png")
            images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 12, "height": 8})
            # The last COCO image is an intentional negative. Unlisted files are ignored.
            if image_id != count:
                annotations.append({"id": image_id, "image_id": image_id, "category_id": 9,
                                    "bbox": [4, 2, 6, 4]})
            group_rows.append({"image_id": image_id, "group_id": f"flight-{image_id}"})
        self.annotations.write_text(json.dumps({"images": images, "annotations": annotations,
                                               "categories": [{"id": 9, "name": "litter"}]}), encoding="utf-8")
        Image.new("RGB", (12, 8), "black").save(self.images / "unlisted.png")
        if groups:
            path = self.root / "groups.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "group_id"])
                writer.writeheader()
                writer.writerows(group_rows)
            return path
        return None

    def rows(self, output, file="split_manifest.csv"):
        with (output / file).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_clipping_and_yolo_normalization(self):
        box = clip_coco_box([-2, 2, 8, 10], 10, 8, 0)
        self.assertEqual(box, Box(0, 0, 2, 6, 8))
        values = list(map(float, yolo_line(box, 10, 8).split()))
        self.assertEqual(values, [0, 0.3, 0.625, 0.6, 0.75])
        for invalid in ([0, 0, 0, 2], [20, 0, 2, 2], [0, 0, float("nan"), 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                clip_coco_box(invalid, 10, 8, 0)

    def test_group_and_duplicate_isolation_is_deterministic(self):
        groups = self.fixture(duplicate=True)
        # Duplicate images 1 and 2 connect flights A and B. Images 3/4 share them.
        groups.write_text("image_id,group_id\n1,A\n2,B\n3,A\n4,B\n5,C\n6,D\n", encoding="utf-8")
        first, second = self.root / "first", self.root / "second"
        manifest = prepare_dataset(self.annotations, self.images, first, groups, seed=17)
        prepare_dataset(self.annotations, self.images, second, groups, seed=17)
        self.assertEqual((first / "split_manifest.csv").read_bytes(), (second / "split_manifest.csv").read_bytes())
        self.assertEqual((first / "dataset_manifest.json").read_bytes(), (second / "dataset_manifest.json").read_bytes())
        rows = self.rows(first)
        self.assertEqual(len({row["split"] for row in rows[:4]}), 1)
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
        self.assertEqual(rows[1]["duplicate_of"], "1")
        self.assertEqual(manifest["counts"]["duplicates_removed"], 1)
        self.assertEqual(manifest["counts"]["output_images"], 5)
        self.assertEqual(manifest["category_mapping"], {"9": 0})
        for pixel_hash in {row["sha256"] for row in rows}:
            self.assertEqual(len({row["split"] for row in rows if row["sha256"] == pixel_hash}), 1)

    def test_tiling_keeps_intersections_and_negatives(self):
        groups = self.fixture()
        output = self.root / "tiled"
        manifest = prepare_dataset(self.annotations, self.images, output, groups, tile_size=6, overlap=0)
        tiles = self.rows(output, "tiles.csv")
        self.assertEqual(manifest["counts"]["output_images"], 24)
        source_splits = {}
        for tile in tiles:
            source_splits.setdefault(tile["image_id"], set()).add(tile["split"])
            label = output / "labels" / tile["split"] / (Path(tile["output_file"]).stem + ".txt")
            lines = label.read_text().splitlines()
            self.assertEqual(len(lines), 0 if tile["image_id"] == "6" else 1)
            for line in lines:
                _, cx, cy, width, height = map(float, line.split())
                self.assertTrue(0 <= cx <= 1 and 0 <= cy <= 1)
                self.assertTrue(0 < width <= 1 and 0 < height <= 1)
                self.assertGreaterEqual(cx - width / 2, -1e-10)
                self.assertLessEqual(cx + width / 2, 1 + 1e-10)
        self.assertTrue(all(len(splits) == 1 for splits in source_splits.values()))
        self.assertFalse(any("unlisted" in tile["source_file"] for tile in tiles))

    def test_missing_groups_requires_explicit_exploratory_opt_in(self):
        self.fixture(groups=False)
        with self.assertRaisesRegex(ValueError, "allow-image-split"):
            prepare_dataset(self.annotations, self.images, self.root / "blocked")
        manifest = prepare_dataset(self.annotations, self.images, self.root / "allowed", allow_image_split=True)
        self.assertEqual(manifest["split_strategy"], "exploratory_image")
        self.assertIn("NOT prevented", manifest["limitations"][0])

    def test_path_traversal_and_absolute_paths_rejected(self):
        groups = self.fixture()
        original = self.annotations.read_text()
        for path in ("../outside.png", "..\\outside.png", "C:\\outside.png", "/outside.png"):
            with self.subTest(path=path):
                document = json.loads(original)
                document["images"][0]["file_name"] = path
                self.annotations.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unsafe image path"):
                    prepare_dataset(self.annotations, self.images, self.root / "unsafe", groups)
                self.assertFalse((self.root / "unsafe").exists())

    def test_actual_dimensions_and_exif_are_validated(self):
        groups = self.fixture()
        document = json.loads(self.annotations.read_text())
        document["images"][0]["width"] = 13
        self.annotations.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            prepare_dataset(self.annotations, self.images, self.root / "mismatch", groups)
        document["images"][0]["width"] = 12
        self.annotations.write_text(json.dumps(document), encoding="utf-8")
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (12, 8), "blue").save(self.images / "1.png", exif=exif)
        with self.assertRaisesRegex(ValueError, "EXIF orientation"):
            prepare_dataset(self.annotations, self.images, self.root / "oriented", groups)

    def test_invalid_reference_and_incomplete_groups_fail(self):
        groups = self.fixture()
        groups.write_text("image_id,group_id\n1,A\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not cover"):
            prepare_dataset(self.annotations, self.images, self.root / "missing", groups)
        document = json.loads(self.annotations.read_text())
        document["annotations"][0]["category_id"] = 999
        self.annotations.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid image/category reference"):
            prepare_dataset(self.annotations, self.images, self.root / "invalid", allow_image_split=True)

    def test_conflicting_duplicate_annotations_and_nonempty_output_fail(self):
        groups = self.fixture(duplicate=True)
        document = json.loads(self.annotations.read_text())
        document["annotations"][1]["bbox"] = [1, 1, 2, 2]
        self.annotations.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicting annotations"):
            prepare_dataset(self.annotations, self.images, self.root / "conflicts", groups)
        output = self.root / "occupied"
        output.mkdir()
        (output / "existing.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing to mix"):
            prepare_dataset(self.annotations, self.images, output, groups)
        self.assertEqual((output / "existing.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

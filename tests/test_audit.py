import json

from PIL import Image

from src.data.audit import audit


def test_audit_records_broken_images_and_invalid_boxes_without_hiding_them(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 100), "green").save(images / "good.png")
    Image.new("RGB", (100, 100), "blue").save(images / "negative.png")
    (images / "bad.png").write_bytes(b"not an image")
    records = [{"id": i, "file_name": n, "width": 100, "height": 100}
               for i, n in enumerate(["good.png", "negative.png", "bad.png"])]
    document = {"images": records, "categories": [{"id": 0, "name": "rubbish"}],
                "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 30]},
                                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [0, 0, -2, 3]}]}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document))
    result = audit(path, images, tmp_path / "report", samples=2)
    assert result["readable_images"] == 2
    assert result["image_statuses"]["missing_or_corrupt"] == 1
    assert len(result["annotation_issues"]) == 1
    assert result["classes"][0]["objects"] == 1
    assert (tmp_path / "report" / "distributions.png").is_file()

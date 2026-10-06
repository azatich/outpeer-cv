"""Check genuine label mapping, multiple class IDs and application integration."""

from types import SimpleNamespace
import json
from unittest.mock import patch

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from src.data.taco import CLASS_MAPPING, rescale_annotations, subset
from src.evaluation.dataset import read_labels
from src.inference.predict import Detector, Prediction, Detection


def test_taco_maps_real_object_categories_without_inventing_labels():
    categories = [{"id": i, "name": name} for i, name in enumerate(
        [name for names in CLASS_MAPPING.values() for name in names] + ["Plastic bottle cap", "Plastic film"])]
    lookup = {c["name"]: c["id"] for c in categories}
    document = {"categories": categories,
                "images": [{"id": i, "file_name": f"batch_1/{i}.jpg"} for i in range(5)],
                "annotations": [{"id": i, "image_id": i, "category_id": lookup[name], "bbox": [2, 3, 4, 5]}
                                for i, name in enumerate(["Glass bottle", "Single-use carrier bag", "Drink can", "Plastic bottle cap", "Plastic film"])]}
    result = subset(document, background_images=2)
    assert [a["category_id"] for a in result["annotations"]] == [0, 1, 2]
    assert all(a["bbox"] == [2, 3, 4, 5] for a in result["annotations"])
    assert len(result["images"]) == 5
    assert [c["name"] for c in result["categories"]] == ["bottle", "bag", "can"]


def test_multiclass_labels_keep_ids_and_reject_unknown_classes(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n")
    assert [t.class_id for t in read_labels(path, 100, 100, 3)] == [0, 1, 2]
    path.write_text("3 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError):
        read_labels(path, 100, 100, 3)
    path.write_text("1.5 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError):
        read_labels(path, 100, 100, 3)


def test_resizing_scales_boxes_and_polygons_with_the_corresponding_image():
    document = {"images": [{"id": 5, "width": 200, "height": 100}],
                "annotations": [{"image_id": 5, "bbox": [20, 10, 40, 20], "area": 800,
                                 "segmentation": [[20, 10, 60, 10, 60, 30]]}]}
    rescale_annotations(document, {5: (100, 50)})
    assert document["images"][0]["width"] == 100
    assert document["annotations"][0]["bbox"] == [10, 5, 20, 10]
    assert document["annotations"][0]["area"] == 200
    assert document["annotations"][0]["segmentation"] == [[10, 5, 30, 5, 30, 15]]


def test_detector_accepts_type_model_and_exports_zero_counts(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fixture")
    names = {0: "bottle", 1: "bag", 2: "can"}
    detector = Detector(weights, "cpu", model_factory=lambda _: SimpleNamespace(names=names, task="detect"))
    assert detector.names == names
    result = Prediction(Image.new("RGB", (40, 30)), [Detection(2, "can", 0.8, (1, 2, 8, 9))],
                        1.0, "cpu", {"conf": 0.25, "classes": names})
    assert result.to_dict()["counts_by_class"] == {"bottle": 0, "bag": 0, "can": 1}
    assert result.annotated().size == (40, 30)


def test_app_reads_valid_custom_selection_before_missing_model_stop(tmp_path):
    weights = tmp_path / "custom" / "best.pt"
    weights.parent.mkdir(); weights.write_bytes(b"fixture")
    decision = {"name": "custom_types", "weights": str(weights), "conf": 0.4,
                "settings": {"imgsz": 640, "nms_iou": 0.7, "max_det": 300}}
    with patch("app.main.available_models", return_value={}), patch("app.main.load_selection", return_value=decision):
        app = AppTest.from_string("from app.main import main\nmain()").run(timeout=30)
    assert not app.exception
    assert "custom_types" in app.selectbox[0].value
    assert app.slider[0].value == 0.4


def test_multiclass_ui_renders_counts_and_download_formats():
    app = AppTest.from_string('''
from PIL import Image
from src.inference.predict import Prediction, Detection
from app.main import render_result
render_result(Prediction(Image.new("RGB", (40, 30)),
    [Detection(0, "bottle", 0.8, (2, 3, 12, 20)), Detection(2, "can", 0.7, (20, 3, 32, 20))],
    2.0, "cpu", {"conf": 0.4, "classes": {0: "bottle", 1: "bag", 2: "can"}}))
''').run(timeout=30)
    assert not app.exception
    assert [m.value for m in app.metric[-3:]] == ["1", "0", "1"]
    assert len(app.get("download_button")) == 3


def test_app_prefers_comparison_winner_instead_of_alphabetical_report_order(tmp_path):
    import app.main as ui

    original, fine = tmp_path / "original.pt", tmp_path / "fine.pt"
    original.write_bytes(b"fixture"); fine.write_bytes(b"fixture")
    root = tmp_path / "reports/multiclass"
    old_selection = root / "validation/selection.json"
    preferred_selection = root / "finetune/validation/selection.json"
    for path in (old_selection, preferred_selection):
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    (root / "recommended.json").write_text(json.dumps({"selection": str(preferred_selection)}))

    def decision(path):
        weights, confidence = (fine, 0.2) if path == preferred_selection else (original, 0.4)
        return {"name": weights.stem, "weights": str(weights), "conf": confidence,
                "settings": {"imgsz": 640, "nms_iou": 0.7, "max_det": 300}}

    with patch.object(ui, "ROOT", tmp_path), patch.object(ui, "available_models", return_value={"Original": original, "Fine": fine}), patch.object(ui, "load_selection", side_effect=decision):
        app = AppTest.from_string("from app.main import main\nmain()").run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].value == "Fine · выбрана на validation"
    assert app.slider[0].value == 0.2

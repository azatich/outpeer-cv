import csv
import hashlib
import json
from pathlib import Path
import struct

import pytest
from PIL import Image
import yaml

from src.evaluation.dataset import PreparedDataset, Truth, read_labels, write_json
from src.evaluation.evaluate import load_selection
from src.evaluation.metrics import choose_threshold, match, score_threshold
from src.inference.predict import Detection
from src.training.train import sha256


def detection(box=(0, 0, 10, 10), confidence=0.9, cls=0):
    return Detection(cls, "rubbish", confidence, box)


def test_matching_counts_duplicate_false_positive_and_missing_truth():
    truth = [Truth(0, (0, 0, 10, 10)), Truth(0, (20, 20, 30, 30))]
    result = match([detection(), detection(confidence=0.7), detection((40, 40, 50, 50))], truth)
    assert result == {"tp": [0], "fp": [2, 1], "fn": [1]}
    metrics = score_threshold([[detection(), detection(confidence=0.2)]], [truth], 0.5, 0.5)
    assert metrics["precision"] == 1 and metrics["recall"] == 0.5


def test_wrong_class_and_insufficient_overlap_do_not_match():
    truth = [Truth(0, (0, 0, 10, 10))]
    assert match([detection(cls=1)], truth)["tp"] == []
    assert match([detection((8, 8, 18, 18))], truth)["fn"] == [0]


def test_empty_predictions_and_validation_threshold_tie_rule():
    assert score_threshold([[]], [[]], 0.5, 0.5)["f1"] == 0
    predictions = [[detection(), detection((20, 20, 30, 30), 0.2)]]
    best, curve = choose_threshold(predictions, [[Truth(0, (0, 0, 10, 10))]], [0.1, 0.4, 0.8])
    assert best["conf"] == 0.8 and best["f1"] == 1
    assert len(curve) == 3


@pytest.fixture
def dataset_fixture(tmp_path):
    root = tmp_path / "data"; reference = tmp_path / "reference"
    reference.mkdir(); root.mkdir()
    splits, tiles = [], []
    for i, split in enumerate(("train", "val", "test")):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        image = Image.new("RGB", (20, 10), (i*50, 0, 0))
        relative = f"images/{split}/image_{i}.png"
        image.save(root / relative)
        (root / "labels" / split / f"image_{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        splits.append(dict(image_id=str(i), file_name=f"image_{i}.png", group_id=split, split=split,
                           sha256=hashlib.sha256(struct.pack(">II", *image.size)+image.tobytes()).hexdigest(), duplicate_of=""))
        tiles.append(dict(image_id=str(i), source_file=f"image_{i}.png", split=split, output_file=relative,
                          x="0", y="0", width="20", height="10", objects="1"))
    manifest = dict(source_annotations_sha256="fixture", seed=42, categories=[{"id": 0, "name": "rubbish"}], image_policy={}, tiling={"enabled": False})
    for directory in (root, reference):
        write_json(directory / "dataset_manifest.json", manifest)
        for filename, rows in (("split_manifest.csv", splits), ("tiles.csv", tiles)):
            with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
                writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {0: "rubbish"}, **{s: f"images/{s}" for s in ("train", "val", "test")}}))
    return root, reference


def test_dataset_identity_and_validation_does_not_decode_test(dataset_fixture):
    root, reference = dataset_fixture
    dataset = PreparedDataset(root / "data.yaml", reference)
    identity = dataset.identity
    (root / "images/test/image_2.png").write_bytes(b"not decoded during validation")
    assert len(dataset.load_split("val")) == 1
    assert "test" not in dataset.resolved_yaml("val")
    (root / "labels/test/image_2.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    assert PreparedDataset(root / "data.yaml", reference).identity != identity


def test_dataset_rejects_changed_split_pixels_and_missing_labels(dataset_fixture):
    root, reference = dataset_fixture
    Image.new("RGB", (20, 10), "green").save(root / "images/val/image_1.png")
    with pytest.raises(ValueError, match="pixels differ"):
        PreparedDataset(root / "data.yaml", reference).load_split("val")
    (root / "labels/val/image_1.txt").unlink()
    with pytest.raises(FileNotFoundError):
        PreparedDataset(root / "data.yaml", reference)


def test_prepared_type_dataset_preserves_class_ids_and_names(dataset_fixture):
    root, reference = dataset_fixture
    names = {0: "bottle", 1: "bag", 2: "can"}
    for directory in (root, reference):
        manifest = json.loads((directory / "dataset_manifest.json").read_text())
        manifest["categories"] = [{"id": class_id, "name": name} for class_id, name in names.items()]
        write_json(directory / "dataset_manifest.json", manifest)
    config = yaml.safe_load((root / "data.yaml").read_text())
    config["names"] = names
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    (root / "labels/val/image_1.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    (root / "labels/test/image_2.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    dataset = PreparedDataset(root / "data.yaml", reference)
    assert dataset.resolved_yaml("val")["names"] == names
    assert dataset.load_split("val")[0].truth[0].class_id == 1
    assert dataset.load_split("test")[0].truth[0].class_id == 2


def test_evaluation_rejects_a_model_with_different_class_meanings(dataset_fixture, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.evaluation import evaluate
    root, reference = dataset_fixture
    dataset = PreparedDataset(root / "data.yaml", reference)
    monkeypatch.setattr(evaluate, "Detector", lambda *args: SimpleNamespace(names={0: "bottle", 1: "bag", 2: "can"}))
    output = tmp_path / "wrong_model"
    with pytest.raises(ValueError, match="classes do not match"):
        evaluate.evaluate_model({"weights": "fixture.pt"}, dataset, dataset.load_split("val"),
                                {"device": "cpu"}, output, split="val")
    assert not output.exists()


@pytest.mark.parametrize("label", ["0 nan 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1", "0 0.9 0.5 0.8 0.8", "0 0.5 0.5 -1 0.5"])
def test_invalid_ground_truth_is_never_silently_ignored(tmp_path, label):
    path = tmp_path / "label.txt"; path.write_text(label)
    with pytest.raises(ValueError):
        read_labels(path, 100, 100)


def test_selection_rejects_threshold_weight_and_report_changes(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"original")
    decision = {"schema_version": 1, "selected_on": "val", "conf": 0.3, "weights": str(weights), "weights_sha256": sha256(weights)}
    report = tmp_path / "comparison.json"; write_json(report, {"selection": decision})
    selected = {**decision, "validation_report": str(report), "validation_report_sha256": sha256(report)}
    path = tmp_path / "selection.json"; write_json(path, selected)
    assert load_selection(path)["conf"] == 0.3
    write_json(path, {**selected, "conf": 0.8})
    with pytest.raises(ValueError, match="settings have changed"):
        load_selection(path)
    write_json(path, selected); weights.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Weights have changed"):
        load_selection(path)
    weights.write_bytes(b"original"); write_json(report, {"selection": {**decision, "conf": 0.8}})
    with pytest.raises(ValueError, match="report has changed"):
        load_selection(path)


def test_b_training_config_matches_actual_a_baseline():
    root = Path(__file__).resolve().parents[1]
    b = yaml.safe_load((root / "configs/b/yolov8s_baseline.yaml").read_text())
    a = json.loads((root / "reports/a/experiments/a_n_baseline_20/metadata.json").read_text())["effective_config"]
    assert b["train"] == a["train"]
    assert b["augmentation"] == a["augmentation"] and b["seed"] == a["seed"]
    assert b["model"] == "yolov8s.pt"


def test_test_evaluation_requires_fixed_threshold_and_never_searches(dataset_fixture, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.evaluation import evaluate
    root, reference = dataset_fixture
    dataset = PreparedDataset(root / "data.yaml", reference)
    samples = dataset.load_split("test")
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fixture")
    spec = {"name": "fixture", "weights": str(weights)}
    config = {"device": "cpu", "imgsz": 640, "nms_iou": 0.7, "max_det": 300,
              "confidence_grid": [0.1, 0.5], "match_iou": 0.5, "examples_per_kind": 1, "benchmark_conf": 0.25}
    with pytest.raises(ValueError, match="previously frozen"):
        evaluate.evaluate_model(spec, dataset, samples, config, tmp_path / "refused", split="test")
    assert not (tmp_path / "refused").exists()
    calls = []
    class FakeDetector:
        def __init__(self, path, device):
            self.weights = Path(path); self.device = device
            self.names = {0: "rubbish"}
            self.model = SimpleNamespace(val=lambda **kw: SimpleNamespace(box=SimpleNamespace(
                mp=1, mr=1, map50=1, map=1, ap_class_index=[0], p=[1], r=[1], ap50=[1], ap=[1])))
        def predict(self, image, **kwargs):
            calls.append(kwargs["conf"])
            return SimpleNamespace(detections=[detection((5, 2.5, 15, 7.5))])
    monkeypatch.setattr(evaluate, "Detector", FakeDetector)
    monkeypatch.setattr(evaluate, "benchmark", lambda *args: {})
    monkeypatch.setattr(evaluate, "choose_threshold", lambda *args: pytest.fail("Test must not select a threshold"))
    result = evaluate.evaluate_model(spec, dataset, samples, config, tmp_path / "test_result", split="test", fixed_conf=0.3)
    assert calls == [0.3] and result["confidence_curve"] == []
    assert result["operating_point"]["tp"] == 1

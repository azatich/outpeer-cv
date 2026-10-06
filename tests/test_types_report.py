"""A comparison report must display the selected model's metrics and provenance."""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image

from scripts import build_types_report as builder
from src.inference.predict import Detection, Prediction
from src.training.train import sha256


def test_report_uses_winner_metrics_when_baseline_is_first(tmp_path, monkeypatch):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    reports = tmp_path / "reports/multiclass/finetune"
    names = {0: "bottle", 1: "bag", 2: "can"}
    op = dict(conf=0.2, precision=0.7, recall=0.6, f1=0.64, tp=3, fp=1, fn=2)

    def row(name, ap50, ap95):
        metrics = dict(precision=0.7, recall=0.6, mAP50=ap50, mAP50_95=ap95)
        return dict(name=name, weights=f"models/b/{name}/best.pt", metrics=metrics, operating_point=op,
                    class_metrics={name: metrics for name in names.values()},
                    class_operating_points={name: op for name in names.values()})

    save(reports / "validation/comparison.json", {"models": [row("baseline", 0.2, 0.1), row("fine", 0.8, 0.6)]})
    selection_path = reports / "validation/selection.json"
    save(selection_path, {})
    selection = dict(name="fine", weights="models/b/fine/best.pt", conf=0.2,
                     settings=dict(imgsz=640, nms_iou=0.7, max_det=300, reference="reports/multiclass/dataset"))
    save(reports / "test/comparison.json", dict(selection_sha256=sha256(selection_path),
         models=[row("fine", 0.75, 0.55)], settings=dict(device="cpu", data="data/data.yaml")))
    source = tmp_path / "data/images/sample.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(source)
    detections = [Detection(i, name, 0.9, (2, 3, 12, 20)) for i, name in names.items()]
    truth = [dict(class_id=d.class_id, xyxy=d.xyxy) for d in detections]
    save(reports / "test/fine/per_image.json", [dict(image="images/sample.png",
         detections=[dict(class_id=d.class_id, class_name=d.class_name, confidence=d.confidence, xyxy=d.xyxy) for d in detections], truth=truth)])
    save(tmp_path / "reports/multiclass/dataset/SUMMARY.json", dict(images=dict(by_split={
         "train": dict(images=10, objects=12, negative_images=2)})))
    train = dict(imgsz=640, batch=2, optimizer="AdamW", lr0=0.0002)
    save(tmp_path / "models/b/fine/metadata.json", dict(effective_config=dict(
         model="models/b/baseline/best.pt", seed=42, train_arguments=train, runtime=dict(amp=False))))
    (tmp_path / "models/b/fine/best.pt").write_bytes(b"fixture")
    save(tmp_path / "models/b/baseline/metadata.json", {})
    (tmp_path / "models/b/baseline/best.pt").write_bytes(b"parent-fixture")
    history = tmp_path / "runs/b/fine/results.csv"
    history.parent.mkdir(parents=True)
    history.write_text("epoch\n1\n2\n")
    detector = SimpleNamespace(names=names, predict=lambda *a, **k: Prediction(
        Image.new("RGB", (40, 30)), detections, 1.0, "cpu", {"conf": 0.2, "classes": names}))
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "local_path", lambda p: tmp_path / p)
    monkeypatch.setattr(builder, "load_selection", lambda p: selection)
    monkeypatch.setattr(builder, "Detector", lambda *a: detector)
    builder.build(reports, archive_name="fine_experiment")

    text = (reports / "RESULTS.md").read_text(encoding="utf-8")
    assert "| validation | 0.8000 | 0.6000 |" in text
    assert "| test | 0.7500 | 0.5500 |" in text
    assert "2 эпох, 640×640, batch=2" in text
    assert "models/b/baseline/best.pt" in text
    with zipfile.ZipFile(tmp_path / "models/b/fine_experiment_handoff.zip") as archive:
        assert archive.testzip() is None
        assert "models/b/fine/best.pt" in archive.namelist()
        assert "models/b/baseline/best.pt" in archive.namelist()
        assert archive.read("reports/multiclass/finetune/training/fine/results.csv") == history.read_bytes()
        assert len(archive.namelist()) == len(set(archive.namelist()))

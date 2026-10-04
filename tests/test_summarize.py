import json

import pytest

from src.training.summarize import summarize


def test_comparison_rejects_different_dataset_and_preserves_real_metrics(tmp_path):
    directories = []
    for name in ("baseline", "augmented"):
        directory = tmp_path / name
        directory.mkdir()
        document = {
            "status": "complete", "name": name, "model": "yolov8n.pt", "seed": 42, "image_size": 640,
            "classes": {"0": "rubbish"}, "provenance": {"dataset_manifest_sha256": "same",
                "data_yaml_sha256": "same_yaml", "pretrained_weights_sha256": "same_weights"},
            "effective_config": {"train": {"epochs": 20}, "runtime": {"amp": False},
                "augmentation": {"degrees": 0 if name == "baseline" else 180}},
            "validation": {"split": "val", "metrics": {"metrics/precision(B)": 0.4, "metrics/recall(B)": 0.5,
                "metrics/mAP50(B)": 0.3, "metrics/mAP50-95(B)": 0.2}},
        }
        (directory / "metadata.json").write_text(json.dumps(document))
        directories.append(directory)
    rows = summarize(directories, tmp_path / "report")
    assert rows[0]["mAP50"] == 0.3
    assert (tmp_path / "report/comparison.png").exists()
    for key in ("data_yaml_sha256", "pretrained_weights_sha256"):
        original = document["provenance"][key]
        document["provenance"][key] = "different"
        (directories[1] / "metadata.json").write_text(json.dumps(document))
        with pytest.raises(ValueError, match=key):
            summarize(directories, tmp_path / "invalid")
        document["provenance"][key] = original
    document["effective_config"]["runtime"]["amp"] = True
    (directories[1] / "metadata.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="runtime AMP"):
        summarize(directories, tmp_path / "invalid")
    document["effective_config"]["runtime"]["amp"] = False
    document["provenance"]["dataset_manifest_sha256"] = "different"
    (directories[1] / "metadata.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="different datasets"):
        summarize(directories, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()

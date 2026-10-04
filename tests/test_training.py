"""Training contract checks use a fake model, without GPU or weight downloads."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from src.training.train import ConfigurationError, prepare_run, run_training, sha256


REPO = Path(__file__).resolve().parents[1]


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with (REPO / "configs" / "a" / "smoke.yaml").open(encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.config["data"] = "dataset/data.yaml"
        self.data_dir = self.root / "dataset"
        for split in ("train", "val", "test"):
            images = self.data_dir / "images" / split
            images.mkdir(parents=True)
            # The wrapper verifies paths; image decoding belongs to preparation/YOLO.
            (images / "image.jpg").write_bytes(b"fixture")
        (self.data_dir / "data.yaml").write_text(yaml.safe_dump({
            "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {0: "litter"},
        }), encoding="utf-8")
        self.manifest = {
            "schema_version": 1, "categories": [{"source_id": 5, "id": 0, "name": "litter"}],
            "split_strategy": "exploratory_image", "limitations": ["Flight IDs unavailable"],
        }
        (self.data_dir / "dataset_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.config_path = self.root / "config.yaml"
        self.write_config()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def plan(self, **overrides):
        return prepare_run(self.config_path, {"device": "cpu", **overrides}, workspace=self.root)

    def test_preflight_resolves_yaml_root_and_has_no_runtime_imports_or_writes(self):
        with patch.dict("sys.modules", {"torch": None, "ultralytics": None}):
            plan = self.plan(epochs=2, batch=1)
        self.assertEqual(plan["dataset"]["path"], str(self.data_dir.resolve()))
        self.assertEqual(plan["dataset"]["train"], str((self.data_dir / "images/train").resolve()))
        self.assertNotIn("test", plan["dataset"])
        self.assertEqual(plan["config"]["train"]["epochs"], 2)
        self.assertEqual(plan["manifest"]["limitations"], self.manifest["limitations"])
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "models").exists())

    def test_rejects_unsafe_name_invalid_probability_and_unknown_training_keys(self):
        original = deepcopy(self.config)
        for mutation in ("unsafe_name", "invalid_probability", "unknown_key", "bad_image_size"):
            with self.subTest(mutation=mutation):
                self.config = deepcopy(original)
                if mutation == "unsafe_name":
                    self.config["name"] = "../overwritten"
                elif mutation == "invalid_probability":
                    self.config["augmentation"]["flipud"] = 1.2
                elif mutation == "unknown_key":
                    self.config["train"]["split"] = "test"
                else:
                    self.config["train"]["imgsz"] = 319
                self.write_config()
                with self.assertRaises(ConfigurationError):
                    self.plan()

    def test_rejects_class_mapping_drift_and_existing_output(self):
        self.manifest["categories"][0]["name"] = "different"
        manifest_path = self.data_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "classes do not match"):
            self.plan()
        self.manifest["categories"][0]["name"] = "litter"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        plan = self.plan()
        plan["run_dir"].mkdir(parents=True)
        marker = plan["run_dir"] / "do-not-change.txt"
        marker.write_text("existing work", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "already exists"):
            self.plan()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing work")

    def test_success_exports_usable_weights_and_complete_validation_provenance(self):
        plan = self.plan()
        received = {}

        class FakeYOLO:
            def __init__(self, path):
                received["model"] = path

            def train(self, **kwargs):
                received.update(kwargs)
                run = Path(kwargs["project"]) / kwargs["name"]
                self.trainer = SimpleNamespace(save_dir=run)
                (run / "weights").mkdir()
                (run / "weights/best.pt").write_bytes(b"fake-checkpoint")
                return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.25, "fitness": 0.2})

        with patch.dict("os.environ", {}):
            metadata = run_training(plan, model_factory=FakeYOLO)
        handoff = plan["handoff_dir"]
        self.assertEqual((handoff / "best.pt").read_bytes(), b"fake-checkpoint")
        self.assertEqual(metadata["weights_sha256"], sha256(handoff / "best.pt"))
        self.assertEqual(metadata["validation"]["split"], "val")
        self.assertFalse(metadata["validation"]["test_evaluated"])
        self.assertEqual(metadata["limitations"], self.manifest["limitations"])
        self.assertEqual(metadata["classes"], {"0": "litter"})
        self.assertEqual(received["split"], "val")
        self.assertTrue(received["deterministic"])
        self.assertEqual(json.loads((plan["run_dir"] / "status.json").read_text())["status"], "complete")
        self.assertEqual(json.loads((handoff / "metadata.json").read_text())["provenance"]["data_yaml_sha256"], sha256(self.data_dir / "data.yaml"))
        for filename in ("classes.json", "effective_config.yaml", "dataset_manifest.json", "environment.json", "validation_metrics.json"):
            self.assertTrue((handoff / filename).is_file(), filename)

    def test_failure_is_recorded_and_never_presents_a_successful_handoff(self):
        plan = self.plan()

        def unavailable_model(_path):
            raise RuntimeError("synthetic CUDA failure")

        with patch.dict("os.environ", {}):
            with self.assertRaisesRegex(RuntimeError, "synthetic CUDA failure"):
                run_training(plan, model_factory=unavailable_model)
        status = json.loads((plan["run_dir"] / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_type"], "RuntimeError")
        self.assertFalse(plan["handoff_dir"].exists())
        self.assertTrue((plan["run_dir"] / "effective_config.yaml").is_file())

    def test_aug_config_changes_only_documented_experimental_factors(self):
        configs = []
        for filename in ("yolov8n_baseline.yaml", "yolov8n_aug.yaml"):
            with (REPO / "configs/a" / filename).open(encoding="utf-8") as handle:
                configs.append(yaml.safe_load(handle))
        baseline, augmented = configs
        self.assertNotEqual(baseline.pop("name"), augmented.pop("name"))
        for field in ("degrees", "flipud"):
            self.assertNotEqual(baseline["augmentation"].pop(field), augmented["augmentation"].pop(field))
        self.assertEqual(baseline, augmented)


if __name__ == "__main__":
    unittest.main()

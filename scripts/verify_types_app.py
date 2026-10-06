"""Verify a frozen type checkpoint in Streamlit with a real image and inference."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from streamlit.testing.v1 import AppTest
import app.main as ui
from src.evaluation.dataset import read_json
from src.evaluation.evaluate import load_selection
from src.inference.predict import local_path


def verify(reports: Path, *, skip_tests: bool = False) -> dict:
    torch.set_num_threads(2)
    selection = load_selection(reports / "validation/selection.json")
    rows = read_json(reports / f"test/{selection['name']}/per_image.json")
    sample = max(rows, key=lambda row: (row["tp"], len({d["class_id"] for d in row["detections"]})))
    assert sample["tp"] > 0, "No correct test detection available for the UI check"
    photo = local_path(selection["settings"]["data"]).parent / sample["image"]
    weights = local_path(selection["weights"])
    raw_label = next((label for label, path in ui.available_models().items() if path.resolve() == weights), selection["name"])
    with patch.object(ui.st, "file_uploader", return_value=BytesIO(photo.read_bytes())):
        app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=90)
        assert not app.exception, [e.message for e in app.exception]
        option = next(label for label in app.selectbox[0].options if label.removesuffix(" · выбрана на validation") == raw_label)
        app.selectbox[0].set_value(option).run(timeout=90)
        assert app.slider[0].value == selection["conf"]
        app.selectbox[1].set_value("cpu").run(timeout=90)
        app.button[0].click().run(timeout=90)
        assert not app.exception, [e.message for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        payload = app.session_state["prediction"].to_dict()
        assert local_path(payload["weights"]) == weights
        assert payload["classes"] == {0: "bottle", 1: "bag", 2: "can"}
        assert set(payload["counts_by_class"]) == {"bottle", "bag", "can"}
        assert sum(payload["counts_by_class"].values()) == payload["count"] > 0
        assert len(app.metric) == 6 and len(app.get("download_button")) == 3
        for d in payload["detections"]:
            x1, y1, x2, y2 = d["xyxy"]
            assert 0 <= x1 < x2 <= payload["width"] and 0 <= y1 < y2 <= payload["height"]
        app.slider[0].set_value(min(0.99, selection["conf"] + 0.10)).run(timeout=90)
        assert not app.exception and len(app.metric) == 0
        result = {"app_test_method": "Streamlit AppTest with actual local checkpoint/image; only file upload transport patched",
                  "browser_test": False, "device": "cpu", "model": selection["name"],
                  "image": sample["image"], "classes": payload["classes"],
                  "counts_by_class": payload["counts_by_class"], "three_download_formats": "passed",
                  "stale_result_reset": "passed", "exceptions": 0}
    if not skip_tests:
        check = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        count = re.search(r"(\d+) passed", check.stdout)
        assert count, check.stdout
        result["tests_passed"] = int(count.group(1))
        result["pytest_output"] = check.stdout.strip()
        (reports / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/multiclass"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    verify(local_path(args.reports), skip_tests=args.skip_tests)

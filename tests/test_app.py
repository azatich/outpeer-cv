from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_explains_missing_model():
    with patch("app.main.available_models", return_value={}), patch("app.main.load_selection", side_effect=FileNotFoundError("No installed model")):
        app = AppTest.from_string("from app.main import main\nmain()").run(timeout=30)
    assert not app.exception
    assert "best.pt" in app.warning[0].value


def test_app_shows_upload_and_clears_stale_result(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fixture")
    with patch("app.main.available_models", return_value={"Fixture": weights}), patch("app.main.load_selection", side_effect=ValueError("No frozen selection in this fixture")):
        app = AppTest.from_string("from app.main import main\nmain()").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Поиск мусора на фотографии"
    assert app.selectbox[0].value == "Fixture"


def test_empty_result_downloads_and_zero_count():
    app = AppTest.from_string('''
from PIL import Image
from src.inference.predict import Prediction
from app.main import render_result
render_result(Prediction(Image.new("RGB", (40, 30)), [], 2.0, "cpu", {"conf": 0.25}))
''').run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "0"
    assert "не гарантирует" in app.info[0].value

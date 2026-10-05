from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from src.inference.predict import Detection, Detector, Prediction, load_image, validate_settings


class Array:
    def __init__(self, values):
        self.values = values
    def cpu(self):
        return self
    def tolist(self):
        return self.values


def test_inference_preserves_rgb_coordinates_and_settings(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fixture")
    received = {}
    class Model:
        names = {0: "rubbish"}
        task = "detect"
        def __init__(self, path):
            pass
        def predict(self, **kwargs):
            received.update(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array([[1, 2, 7, 8]]), conf=Array([0.8]), cls=Array([0])))]
    result = Detector(weights, "cpu", model_factory=Model).predict(Image.new("RGB", (10, 12), "red"), conf=0.4)
    assert received["source"].getpixel((0, 0)) == (255, 0, 0)
    assert received["conf"] == 0.4 and received["rect"] is False
    assert result.to_dict()["count"] == 1
    assert result.detections[0].xyxy == (1, 2, 7, 8)
    assert result.annotated().size == (10, 12)


def test_missing_weights_fail_without_downloading(tmp_path):
    def factory(_):
        pytest.fail("Must not load or download missing weights")
    with pytest.raises(FileNotFoundError):
        Detector(tmp_path / "missing.pt", "cpu", model_factory=factory)


def test_wrong_class_mapping_rejected(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="rubbish"):
        Detector(weights, "cpu", model_factory=lambda _: SimpleNamespace(names={0: "person"}, task="detect"))


@pytest.mark.parametrize("kwargs", [{"conf": float("nan")}, {"conf": 0}, {"imgsz": 641}, {"max_det": 0}, {"iou": 1.1}])
def test_invalid_prediction_settings(kwargs):
    settings = dict(imgsz=640, conf=0.25, iou=0.7, max_det=300); settings.update(kwargs)
    with pytest.raises(ValueError):
        validate_settings(**settings)


def test_exif_display_orientation_and_invalid_upload():
    image = Image.new("RGB", (20, 10), "blue")
    exif = Image.Exif(); exif[274] = 6
    buffer = BytesIO(); image.save(buffer, format="JPEG", exif=exif)
    assert load_image(buffer.getvalue()).size == (10, 20)
    with pytest.raises(ValueError, match="Cannot read"):
        load_image(b"not an image")


def test_no_detections_produce_valid_empty_result():
    result = Prediction(Image.new("RGB", (10, 10)), [], 1.0, "cpu", {})
    assert result.to_dict()["count"] == 0
    assert result.annotated().tobytes() == result.image.tobytes()


def test_large_upload_rejected_before_conversion():
    image = Image.new("RGB", (20, 10))
    buffer = BytesIO(); image.save(buffer, format="PNG")
    with pytest.raises(ValueError, match="exceeds"):
        load_image(buffer.getvalue(), max_pixels=100)

"""Run a local litter checkpoint on one photo; export PNG and JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models/a/a_n_baseline_20/best.pt"
TYPE_WEIGHTS = ROOT / "models/b/taco_n_types_30/best.pt"
SUPPORTED_CLASS_MAPS = ({0: "rubbish"}, {0: "bottle", 1: "bag", 2: "can"})
CLASS_LABELS = {"rubbish": "Мусор", "bottle": "Бутылка", "bag": "Пакет", "can": "Банка"}
CLASS_COLORS = {"rubbish": "#00c28a", "bottle": "#0099ff", "bag": "#ffba08", "can": "#e85aad"}


def validate_classes(names: dict[int, str]) -> None:
    if names not in SUPPORTED_CLASS_MAPS:
        raise ValueError(f"Expected 0=rubbish or bottle/bag/can detection classes; received {names}")


def local_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def runtime_environment() -> None:
    for variable, folder in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        target = ROOT / ".cache" / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(target))


def resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "mps"} and not re.fullmatch(r"\d+", device):
        raise ValueError("device must be auto, cpu, mps, or a GPU index")
    if device == "auto":
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def validate_settings(imgsz: int, conf: float, iou: float, max_det: int) -> None:
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    for name, value in (("conf", conf), ("iou", iou)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if isinstance(max_det, bool) or not isinstance(max_det, int) or max_det < 1:
        raise ValueError("max_det must be a positive integer")


def load_image(source: str | Path | bytes | Image.Image, *, max_pixels: int | None = None) -> Image.Image:
    """Return detached RGB pixels in display orientation; never retain an upload handle."""
    try:
        if isinstance(source, Image.Image):
            if max_pixels is not None and source.width * source.height > max_pixels:
                raise ValueError(f"Image exceeds {max_pixels:,} pixels")
            return ImageOps.exif_transpose(source).convert("RGB").copy()
        handle = BytesIO(source) if isinstance(source, bytes) else local_path(source)
        with Image.open(handle) as image:
            if max_pixels is not None and image.width * image.height > max_pixels:
                raise ValueError(f"Image exceeds {max_pixels:,} pixels")
            image.seek(0)
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image: {exc}") from exc


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class Prediction:
    image: Image.Image
    detections: list[Detection]
    elapsed_ms: float
    device: str
    settings: dict
    stage_ms: dict | None = None

    def to_dict(self) -> dict:
        names = self.settings.get("classes", {}).values()
        counts = {name: 0 for name in names}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return {
            "width": self.image.width, "height": self.image.height,
            "count": len(self.detections), "detections": [asdict(d) for d in self.detections],
            "counts_by_class": counts,
            "predict_ms": self.elapsed_ms, "device": self.device, "stage_ms": self.stage_ms, **self.settings,
            "coordinate_system": "xyxy pixels of EXIF-oriented RGB image",
            "timing_scope": "predict + CPU box extraction; excludes load/decode/drawing; may include first-call warmup",
        }

    def annotated(self) -> Image.Image:
        output = self.image.copy()
        draw = ImageDraw.Draw(output)
        width = max(2, round(min(output.size) / 400))
        for item in self.detections:
            color = CLASS_COLORS.get(item.class_name, "#00c28a")
            draw.rectangle(item.xyxy, outline=color, width=width)
            x, y = item.xyxy[:2]
            text = f"{item.class_name} {item.confidence:.0%}"
            position = (x, max(0, y - 15))
            draw.rectangle(draw.textbbox(position, text), fill="#12362d")
            draw.text(position, text, fill="white")
        return output


class Detector:
    """One model per instance. A lock makes Streamlit's shared resource safe to call."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "auto", *, model_factory: Any = None):
        self.weights = local_path(weights)
        if not self.weights.is_file() or self.weights.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing trained weights: {self.weights}. Restore models/a from participant_a_handoff.zip; see reports/b/README.md."
            )
        if self.weights.suffix.lower() != ".pt":
            raise ValueError("Expected a local .pt checkpoint")
        runtime_environment()
        self.device = resolve_device(device)
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        try:
            self.model = model_factory(str(self.weights))
        except Exception as exc:
            raise RuntimeError(f"Cannot load checkpoint {self.weights.name}: {exc}") from exc
        names = self.model.names
        self.names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        validate_classes(self.names)
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError("Expected an object detection checkpoint")
        self._lock = threading.Lock()

    def predict(self, source: str | Path | bytes | Image.Image, *, imgsz: int = 640,
                conf: float = 0.25, iou: float = 0.7, max_det: int = 300) -> Prediction:
        validate_settings(imgsz, conf, iou, max_det)
        pixels = load_image(source)
        with self._lock:
            start = perf_counter()
            result = self.model.predict(
                source=pixels, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                device=self.device, verbose=False, save=False, half=False,
                augment=False, rect=False, batch=1,
            )[0]
            boxes = result.boxes
            detections = []
            if boxes is not None:
                for xyxy, score, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                    class_id = int(class_id)
                    detections.append(Detection(class_id, self.names[class_id], float(score), tuple(map(float, xyxy))))
            elapsed = (perf_counter() - start) * 1000
        return Prediction(pixels, detections, elapsed, self.device,
                          {"imgsz": imgsz, "conf": conf, "nms_iou": iou, "max_det": max_det, "weights": str(self.weights), "classes": self.names},
                          {k: float(v) for k, v in (getattr(result, "speed", None) or {}).items() if v is not None})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True, help="New directory for annotated.png and detections.json")
    args = parser.parse_args(argv)
    try:
        output = local_path(args.output)
        if output.exists():
            raise ValueError(f"Output already exists: {output}; choose a new directory")
        result = Detector(args.weights, args.device).predict(args.image, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det)
        output.mkdir(parents=True, exist_ok=False)
        result.annotated().save(output / "annotated.png")
        (output / "detections.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Detections: {len(result.detections)}; saved to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

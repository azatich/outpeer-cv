"""Compare checkpoints on validation and freeze a model/threshold for held-out testing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import statistics
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import yaml

from src.evaluation.dataset import PreparedDataset, Sample, read_json, write_json
from src.evaluation.metrics import choose_threshold, match, score_threshold
from src.inference.predict import Detector, ROOT, local_path, load_image, validate_settings
from src.training.train import environment_snapshot, sha256

CONFIG_KEYS = {"data", "reference", "device", "imgsz", "nms_iou", "match_iou", "max_det",
               "confidence_grid", "benchmark_images", "benchmark_conf", "benchmark_repeats", "warmup", "examples_per_kind", "models"}


def relative_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(local_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"Evaluation config must have exactly: {sorted(CONFIG_KEYS)}")
    validate_settings(config["imgsz"], 0.001, config["nms_iou"], config["max_det"])
    validate_settings(config["imgsz"], config["benchmark_conf"], config["nms_iou"], config["max_det"])
    if not 0 < config["match_iou"] <= 1:
        raise ValueError("match_iou must be in (0,1]")
    grid = config["confidence_grid"]
    if not isinstance(grid, list) or not grid or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.001 <= v <= 1 for v in grid):
        raise ValueError("confidence_grid must contain values in [0.001,1]")
    for key in ("benchmark_images", "benchmark_repeats", "warmup", "examples_per_kind"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("At least one model is required")
    names = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != {"name", "weights"}:
            raise ValueError("Each model requires name and weights")
        if not isinstance(model["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", model["name"]) or model["name"] in names:
            raise ValueError("Model names must be unique, simple directory names")
        names.add(model["name"])
        if not isinstance(model["weights"], str) or not local_path(model["weights"]).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {model['weights']}")
    return config


def synchronize(device: str) -> None:
    import torch
    if device.isdigit():
        torch.cuda.synchronize(int(device))
    elif device == "mps":
        torch.mps.synchronize()


def benchmark(detector: Detector, samples: list[Sample], config: dict, conf: float) -> dict:
    import torch
    indices = np.linspace(0, len(samples)-1, min(len(samples), config["benchmark_images"]), dtype=int)
    chosen = [samples[i] for i in indices]
    pixels = [load_image(sample.image) for sample in chosen]
    kwargs = dict(imgsz=config["imgsz"], conf=conf, iou=config["nms_iou"], max_det=config["max_det"])
    for _ in range(config["warmup"]):
        detector.predict(pixels[0], **kwargs)
    latencies = []
    for _ in range(config["benchmark_repeats"]):
        for image in pixels:
            synchronize(detector.device)
            start = perf_counter()
            detector.predict(image, **kwargs)
            synchronize(detector.device)
            latencies.append((perf_counter()-start)*1000)
    name = torch.cuda.get_device_name(int(detector.device)) if detector.device.isdigit() else platform.processor() or detector.device
    return {"device": detector.device, "device_name": name, "torch": torch.__version__,
            "imgsz": config["imgsz"], "conf": conf, "batch": 1, "precision": "FP32", "rect": False,
            "warmup": config["warmup"], "repeats": config["benchmark_repeats"], "samples": len(latencies),
            "images": [sample.relative for sample in chosen],
            "source_sizes": [[sample.width, sample.height] for sample in chosen],
            "mean_ms": statistics.mean(latencies), "median_ms": statistics.median(latencies),
            "p95_ms": float(np.percentile(latencies, 95)), "measurements_ms": latencies,
            "scope": "RGB image copy + preprocessing + inference + NMS + CPU boxes; synchronized; excludes file decode, model load, drawing and disk writes"}


def error_examples(samples: list[Sample], predictions: list, conf: float, match_iou: float, output: Path, limit: int) -> dict:
    output.mkdir()
    examples = {"tp": [], "fp": [], "fn": []}
    rows = []
    for sample, raw_detections in zip(samples, predictions):
        detections = [d for d in raw_detections if d.confidence >= conf]
        matched = match(detections, sample.truth, match_iou)
        rows.append({"image": sample.relative, **{k: len(v) for k, v in matched.items()},
                     "detections": [asdict(d) for d in detections], "truth": [asdict(t) for t in sample.truth]})
        kinds = [k for k in examples if matched[k] and len(examples[k]) < limit]
        if not kinds:
            continue
        image = load_image(sample.image)
        draw = ImageDraw.Draw(image)
        width = max(3, min(image.size)//250)
        for kind in ("tp", "fp"):
            for i in matched[kind]:
                box = detections[i].xyxy
                color = "#00be70" if kind == "tp" else "#ff4040"
                draw.rectangle(box, outline=color, width=width)
                draw.text((box[0], max(0, box[1]-14)), f"{kind.upper()} {detections[i].confidence:.2f}", fill=color, stroke_width=1, stroke_fill="black")
        for i in matched["fn"]:
            box = sample.truth[i].xyxy
            draw.rectangle(box, outline="#ffb000", width=width)
            draw.text((box[0], max(0, box[1]-14)), "FN (ground truth)", fill="#ffb000", stroke_width=1, stroke_fill="black")
        image.thumbnail((1600, 1600))
        filename = f"{sample.image.stem}.jpg"
        image.save(output / filename, quality=88)
        for kind in kinds:
            examples[kind].append({"image": sample.relative, "file": f"examples/{filename}"})
    write_json(output.parent / "per_image.json", rows)
    write_json(output / "index.json", examples)
    return examples


def evaluate_model(model_spec: dict, dataset: PreparedDataset, samples: list[Sample], config: dict,
                   output: Path, *, split: str, fixed_conf: float | None = None) -> dict:
    if split == "test" and fixed_conf is None:
        raise ValueError("Test requires a previously frozen confidence")
    detector = Detector(model_spec["weights"], str(config["device"]))
    output.mkdir(parents=True, exist_ok=False)
    data_yaml = output / "dataset.resolved.yaml"
    data_yaml.write_text(yaml.safe_dump(dataset.resolved_yaml(split), sort_keys=False), encoding="utf-8")
    # AP uses a low score floor, independently of the application confidence threshold.
    result = detector.model.val(data=str(data_yaml), split=split, imgsz=config["imgsz"],
                                batch=1, device=detector.device, workers=0, conf=0.001,
                                iou=config["nms_iou"], max_det=config["max_det"], half=False,
                                rect=False, augment=False, plots=True, save_json=False,
                                project=str(output), name="ultralytics", exist_ok=False, verbose=False)
    metrics = {"precision": float(result.box.mp), "recall": float(result.box.mr),
               "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map)}
    floor = min(config["confidence_grid"]) if fixed_conf is None else fixed_conf
    predictions = []
    for index, sample in enumerate(samples, 1):
        prediction = detector.predict(sample.image, imgsz=config["imgsz"], conf=floor,
                                      iou=config["nms_iou"], max_det=config["max_det"])
        predictions.append(prediction.detections)
        if index % 40 == 0:
            print(f"{model_spec['name']} {split}: {index}/{len(samples)}", flush=True)
    truths = [sample.truth for sample in samples]
    if fixed_conf is None:
        operating, curve = choose_threshold(predictions, truths, config["confidence_grid"], config["match_iou"])
    else:
        operating = score_threshold(predictions, truths, fixed_conf, config["match_iou"])
        curve = []
    latency = benchmark(detector, samples, config, config["benchmark_conf"])
    examples = error_examples(samples, predictions, operating["conf"], config["match_iou"], output / "examples", config["examples_per_kind"])
    summary = {"name": model_spec["name"], "weights": relative_path(detector.weights),
               "weights_sha256": sha256(detector.weights), "split": split,
               "images": len(samples), "objects": sum(len(s.truth) for s in samples),
               "metrics": metrics, "operating_point": operating, "confidence_curve": curve,
               "latency": latency, "examples": examples,
               "metric_note": "Ultralytics P/R use its own best-F1 curve point; operating_point is micro P/R at the fixed application threshold (greedy class-aware IoU matching)."}
    write_json(output / "summary.json", summary)
    return summary


def write_report(output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["models"]
    lines = [f"# Оценка: {report['split']}", "", f"Устройство: {rows[0]['latency']['device_name']}; imgsz={report['settings']['imgsz']}; batch=1; FP32.", "",
             "| Модель | Precision¹ | Recall¹ | mAP50 | mAP50–95 | Порог | F1² | Среднее, мс | p95, мс |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m, op, lat = row["metrics"], row["operating_point"], row["latency"]
        lines.append(f"| {row['name']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {op['conf']:.2f} | {op['f1']:.4f} | {lat['mean_ms']:.2f} | {lat['p95_ms']:.2f} |")
    lines += ["", "¹ Precision/Recall Ultralytics при её внутренней оптимальной точке кривой. AP считается с conf=0.001.",
              f"² Рабочий порог выбирается только на validation по максимальному micro F1 при IoU={report['settings']['match_iou']}; при равенстве — Precision, затем больший порог.",
              "", f"Время: одинаковый conf={report['settings']['benchmark_conf']} для всех моделей; копирование RGB, preprocessing, inference, NMS и перенос рамок на CPU после прогрева; без чтения файла, рисования и загрузки весов. GPU синхронизирован.",
              "", "## Рабочая точка", "", "| Модель | TP | FP | FN | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        op = row["operating_point"]
        lines.append(f"| {row['name']} | {op['tp']} | {op['fp']} | {op['fn']} | {op['precision']:.4f} | {op['recall']:.4f} |")
    lines += ["", "## Примеры ошибок", "", "Зелёный — TP, красный — FP, жёлтый — FN (рамка разметки). Один кадр может содержать несколько видов ошибок."]
    for row in rows:
        lines += ["", f"### {row['name']}"]
        for kind, examples in row["examples"].items():
            if not examples:
                lines.append(f"\n{kind.upper()}: таких случаев при выбранных настройках нет.")
            for example in examples:
                lines.append(f"\n![{kind.upper()}: {example['image']}]({row['name']}/{example['file']})")
    lines += ["", "## Ограничения", "", "Серии определены по именам файлов; независимость полётов/локаций не подтверждена. Один seed. Нет отдельных чистых сцен. Результаты не подтверждают качество на море, побережье или спутниковых снимках."]
    if report["split"] == "val":
        lines += ["", f"Выбрана **{report['selection']['name']}** по максимальному validation mAP50–95. Настройки зафиксированы в `selection.json`. Test в этом запуске не использовался."]
    else:
        lines += ["", "Модель и порог взяты из зафиксированного решения validation; на test настройки не подбирались."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["name"] for r in rows]
    axes[0].bar(names, [r["metrics"]["mAP50_95"] for r in rows], color="#109c80")
    axes[0].set(title=f"{report['split']}: mAP50-95", ylim=(0, 1))
    axes[1].bar(names, [r["latency"]["mean_ms"] for r in rows], color="#417fc6")
    axes[1].set(title="Warm inference, batch=1", ylabel="ms / image")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)


def run_validation(config: dict, output: Path) -> dict:
    dataset = PreparedDataset(config["data"], config["reference"])
    samples = dataset.load_split("val")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "split": "val"})
    try:
        rows = [evaluate_model(model, dataset, samples, config, output / model["name"], split="val") for model in config["models"]]
        # Deterministic model selection; speed is reported, never silently used to retune on test.
        best = sorted(rows, key=lambda row: (-row["metrics"]["mAP50_95"], row["name"]))[0]
        settings = {key: value for key, value in config.items() if key != "models"}
        selection = {"schema_version": 1, "selected_on": "val", "criterion": "max mAP50_95; tie: model name",
                     "name": best["name"], "weights": best["weights"], "weights_sha256": best["weights_sha256"],
                     "conf": best["operating_point"]["conf"], "dataset_identity": dataset.identity, "settings": settings}
        report = {"schema_version": 1, "split": "val", "test_evaluated": False,
                  "created_at": datetime.now(timezone.utc).isoformat(), "dataset_identity": dataset.identity,
                  "settings": settings, "environment": environment_snapshot(ROOT), "models": rows, "selection": selection}
        write_json(output / "comparison.json", report)
        selection_file = {**selection, "validation_report": relative_path(output / "comparison.json"),
                          "validation_report_sha256": sha256(output / "comparison.json")}
        write_json(output / "selection.json", selection_file)
        write_report(output, report)
        write_json(output / "status.json", {"status": "complete", "split": "val"})
        return report
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def load_selection(path: str | Path) -> dict:
    selection = read_json(local_path(path))
    if selection.get("schema_version") != 1 or selection.get("selected_on") != "val":
        raise ValueError("Expected a frozen validation selection")
    report_path = local_path(selection["validation_report"])
    if sha256(report_path) != selection["validation_report_sha256"]:
        raise ValueError("Validation report has changed since selection")
    stored = {k: v for k, v in selection.items() if k not in {"validation_report", "validation_report_sha256"}}
    if stored != read_json(report_path)["selection"]:
        raise ValueError("Selection settings have changed since validation")
    if sha256(local_path(selection["weights"])) != selection["weights_sha256"]:
        raise ValueError("Weights have changed since validation")
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/b/evaluation.yaml")
    parser.add_argument("--output", default="reports/b/validation")
    parser.add_argument("--device", help="Override evaluation device consistently for every model")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.device is not None:
            config["device"] = args.device
        run_validation(config, local_path(args.output))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

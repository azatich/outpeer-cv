"""Supplement the frozen comparison with alternating model timings and stage breakdowns."""

import argparse
import random
import statistics
from time import perf_counter

import numpy as np

from src.evaluation.dataset import PreparedDataset, read_json
from src.evaluation.evaluate import synchronize
from src.inference.predict import Detector, load_image, local_path
from src.training.train import sha256


def run_benchmark(report_path, output):
    report = read_json(report_path)
    if report["split"] != "val":
        raise ValueError("Benchmark input must be the frozen validation comparison")
    config = report["settings"]
    dataset = PreparedDataset(config["data"], config["reference"])
    if dataset.identity != report["dataset_identity"]:
        raise ValueError("Dataset differs from validation")
    samples = dataset.load_split("val")
    indices = np.linspace(0, len(samples)-1, min(len(samples), config["benchmark_images"]), dtype=int)
    chosen = [samples[i] for i in indices]
    models = {}
    for row in report["models"]:
        if sha256(local_path(row["weights"])) != row["weights_sha256"]:
            raise ValueError(f"Changed weights: {row['name']}")
        models[row["name"]] = Detector(row["weights"], str(config["device"]))
    kwargs = dict(imgsz=config["imgsz"], conf=config["benchmark_conf"], iou=config["nms_iou"], max_det=config["max_det"])
    first = load_image(chosen[0].image)
    for detector in models.values():
        for _ in range(config["warmup"]):
            detector.predict(first, **kwargs)
    records = {name: [] for name in models}
    rng = random.Random(42)
    # Decode one image at a time outside the timer, avoiding a large multi-image RGB cache.
    # Shuffle model order for each image/repeat so CPU cache/thermal drift are shared.
    for repeat in range(config["benchmark_repeats"]):
        for sample in chosen:
            pixels = load_image(sample.image)
            names = list(models); rng.shuffle(names)
            for name in names:
                detector = models[name]
                synchronize(detector.device)
                start = perf_counter()
                result = detector.predict(pixels, **kwargs)
                synchronize(detector.device)
                records[name].append({"image": sample.relative, "repeat": repeat,
                                      "wall_ms": (perf_counter()-start)*1000, **result.stage_ms})
    summary = {}
    for name, rows in records.items():
        summary[name] = {}
        for metric in ("wall_ms", "preprocess", "inference", "postprocess"):
            values = [row[metric] for row in rows]
            summary[name][metric] = {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                                     "p95_ms": float(np.percentile(values, 95))}
    result = {"split": "val", "source_report_sha256": sha256(report_path), "settings": config,
              "device_name": report["models"][0]["latency"]["device_name"],
              "images": [s.relative for s in chosen], "samples_per_model": len(next(iter(records.values()))),
              "method": f"FP32, batch=1; {config['warmup']} warmups; seed-42 shuffled model order per image/repeat; one decoded RGB image at a time; synchronized CUDA; excludes file decode/model load/drawing",
              "stage_source": "Ultralytics Results.speed (ms); wall_ms additionally includes PIL copies, API overhead and CPU box extraction",
              "models": summary, "measurements": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        import json
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/b/validation/comparison.json")
    parser.add_argument("--output", default="reports/b/latency.json")
    args = parser.parse_args()
    run_benchmark(local_path(args.report), local_path(args.output))


if __name__ == "__main__":
    main()

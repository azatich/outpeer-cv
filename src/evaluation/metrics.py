"""Operating-point metrics: confidence-ordered, class-aware, one-to-one IoU matching."""

from __future__ import annotations

from src.evaluation.dataset import Truth
from src.inference.predict import Detection


def iou(a: tuple, b: tuple) -> float:
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match(detections: list[Detection], truth: list[Truth], threshold: float = 0.5) -> dict:
    unmatched = set(range(len(truth)))
    tp, fp = [], []
    for index in sorted(range(len(detections)), key=lambda i: (-detections[i].confidence, i)):
        prediction = detections[index]
        candidates = [(iou(prediction.xyxy, truth[j].xyxy), j) for j in sorted(unmatched)
                      if truth[j].class_id == prediction.class_id]
        overlap, target = max(candidates, default=(0., -1))
        if target >= 0 and overlap >= threshold:
            unmatched.remove(target)
            tp.append(index)
        else:
            fp.append(index)
    return {"tp": tp, "fp": fp, "fn": sorted(unmatched)}


def counts_to_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2*precision*recall / (precision+recall) if precision+recall else 0.}


def score_threshold(predictions: list[list[Detection]], truths: list[list[Truth]], conf: float, match_iou: float) -> dict:
    if len(predictions) != len(truths):
        raise ValueError("Prediction and truth image counts differ")
    totals = dict(tp=0, fp=0, fn=0)
    for detections, truth in zip(predictions, truths):
        result = match([d for d in detections if d.confidence >= conf], truth, match_iou)
        for key in totals:
            totals[key] += len(result[key])
    return {"conf": conf, **counts_to_metrics(**totals)}


def choose_threshold(predictions: list, truths: list, grid: list[float], match_iou: float = 0.5) -> tuple[dict, list[dict]]:
    if not grid or any(not 0 < conf <= 1 for conf in grid):
        raise ValueError("Confidence grid must contain values in (0, 1]")
    scores = [score_threshold(predictions, truths, conf, match_iou) for conf in sorted(set(grid))]
    return max(scores, key=lambda row: (row["f1"], row["precision"], row["conf"])), scores

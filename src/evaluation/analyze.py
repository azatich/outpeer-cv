"""Explain saved validation errors without running inference or changing thresholds."""

import argparse

from src.evaluation.dataset import Truth, read_json, write_json
from src.evaluation.metrics import iou, match
from src.inference.predict import Detection, local_path


def analyze(run):
    report = read_json(run / "comparison.json")
    if report["split"] != "val":
        raise ValueError("Exploratory error analysis uses validation only")
    name = report["selection"]["name"]
    rows = read_json(run / name / "per_image.json")
    from PIL import Image
    data_root = local_path(report["settings"]["data"]).parent
    bins = {label: {"objects": 0, "found": 0} for label in ("short_side_lt_8px", "short_side_8_to_16px", "short_side_ge_16px")}
    false_positives = {"overlap_lt_0.1": 0, "overlap_0.1_to_match_iou": 0, "duplicate_or_assignment_conflict": 0}
    match_iou = report["settings"]["match_iou"]
    for row in rows:
        detections = [Detection(**value) for value in row["detections"]]
        truth = [Truth(**value) for value in row["truth"]]
        matched = match(detections, truth, match_iou)
        with Image.open(data_root / row["image"]) as image:
            scale = report["settings"]["imgsz"] / max(image.size)
        missed = set(matched["fn"])
        for index, target in enumerate(truth):
            x1, y1, x2, y2 = target.xyxy
            short = min(x2-x1, y2-y1)*scale
            category = "short_side_lt_8px" if short < 8 else "short_side_8_to_16px" if short < 16 else "short_side_ge_16px"
            bins[category]["objects"] += 1
            bins[category]["found"] += index not in missed
        for index in matched["fp"]:
            overlap = max((iou(detections[index].xyxy, target.xyxy) for target in truth), default=0)
            category = "overlap_lt_0.1" if overlap < 0.1 else "overlap_0.1_to_match_iou" if overlap < match_iou else "duplicate_or_assignment_conflict"
            false_positives[category] += 1
    for values in bins.values():
        values["recall"] = values["found"]/values["objects"] if values["objects"] else None
    result = {"split": "val", "model": name, "conf": report["selection"]["conf"],
              "match_iou": match_iou, "imgsz": report["settings"]["imgsz"],
              "size_bins": bins, "false_positive_overlap": false_positives,
              "notes": ["Short side is measured after aspect-preserving resize to imgsz, before padding.",
                        "FP overlap groups are geometric descriptions, not proven causes or annotation errors.",
                        "A poorly localized prediction can count as both FP and FN for the same object."]}
    write_json(run / "error_breakdown.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="reports/b/validation")
    args = parser.parse_args()
    import json
    print(json.dumps(analyze(local_path(args.run)), indent=2))


if __name__ == "__main__":
    main()

"""Export comparable completed A experiments; never substitute missing metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(handoffs: list[Path], output: Path) -> list[dict]:
    documents = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in handoffs]
    if len(documents) < 2:
        raise ValueError("Provide at least two completed experiments")
    first = documents[0]
    for document in documents:
        if document.get("status") != "complete" or document["validation"]["split"] != "val":
            raise ValueError("Only completed validation runs can be compared here")
        if document["provenance"]["dataset_manifest_sha256"] != first["provenance"]["dataset_manifest_sha256"]:
            raise ValueError("Experiments used different datasets/splits")
        for key in ("data_yaml_sha256", "pretrained_weights_sha256"):
            if not document["provenance"].get(key) or document["provenance"][key] != first["provenance"].get(key):
                raise ValueError(f"Experiments require matching {key}")
        if document["classes"] != first["classes"]:
            raise ValueError("Experiments used different classes")
        if document["seed"] != first["seed"]:
            raise ValueError("Use the same seed for this paired augmentation experiment")
        if document["effective_config"]["train"] != first["effective_config"]["train"]:
            raise ValueError("Experiments used different training budgets/settings")
        if document["model"] != first["model"]:
            raise ValueError("This report compares augmentation policies of the same architecture")
        runtime = document["effective_config"].get("runtime", {})
        if "amp" not in runtime or runtime["amp"] != first["effective_config"].get("runtime", {}).get("amp"):
            raise ValueError("Experiments require matching actual runtime AMP")
    fields = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
              "mAP50": "metrics/mAP50(B)", "mAP50_95": "metrics/mAP50-95(B)"}
    rows = []
    for document, directory in zip(documents, handoffs):
        metrics = document["validation"]["metrics"]
        if any(key not in metrics for key in fields.values()):
            raise ValueError(f"Missing expected validation metrics in {directory}")
        config = document["effective_config"]
        rows.append({"run": document["name"], "split": "val", "epochs_budget": config["train"]["epochs"],
                     "imgsz": document["image_size"], "seed": document["seed"],
                     **{label: metrics[key] for label, key in fields.items()},
                     "weights": (directory / "best.pt").as_posix()})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps({
        "split": "val", "test_evaluated": False, "results": rows,
        "dataset_manifest_sha256": first["provenance"]["dataset_manifest_sha256"],
        "augmentation_policies": {d["name"]: d["effective_config"]["augmentation"] for d in documents},
        "limitations": first.get("limitations", []) + [
            "Single-seed experiment: differences do not establish statistical significance.",
            "Checkpoint selection used validation; final test evaluation belongs to participant B."],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("mAP50", "mAP50_95")):
        axis.bar([row["run"] for row in rows], [row[metric] for row in rows])
        axis.set(title=f"Validation {metric}", ylim=(0, 1))
        axis.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoffs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("reports/a/experiments"))
    args = parser.parse_args()
    for row in summarize(args.handoffs, args.output):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

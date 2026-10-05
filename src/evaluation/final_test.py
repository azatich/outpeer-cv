"""Evaluate the frozen validation winner once on held-out test; no threshold search."""

import argparse
from datetime import datetime, timezone

from src.evaluation.dataset import PreparedDataset, write_json
from src.evaluation.evaluate import evaluate_model, load_selection, write_report
from src.inference.predict import ROOT, local_path
from src.training.train import environment_snapshot, sha256


def run_test(selection_path, output, *, device=None, data=None):
    selection = load_selection(selection_path)
    config = dict(selection["settings"])
    if device is not None:
        config["device"] = device
    if data is not None:
        config["data"] = data
    dataset = PreparedDataset(config["data"], config["reference"])
    if dataset.identity != selection["dataset_identity"]:
        raise ValueError("Dataset has changed since validation selection")
    samples = dataset.load_split("test")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "split": "test"})
    try:
        row = evaluate_model(selection, dataset, samples, config, output / selection["name"], split="test", fixed_conf=selection["conf"])
        report = {"schema_version": 1, "split": "test", "test_evaluated": True,
                  "created_at": datetime.now(timezone.utc).isoformat(), "settings": config,
                  "dataset_identity": dataset.identity, "selection_sha256": sha256(local_path(selection_path)),
                  "environment": environment_snapshot(ROOT), "models": [row]}
        write_json(output / "comparison.json", report)
        write_report(output, report)
        write_json(output / "status.json", {"status": "complete", "split": "test"})
        return report
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="reports/b/validation/selection.json")
    parser.add_argument("--output", default="reports/b/test")
    parser.add_argument("--device")
    parser.add_argument("--data", help="Relocated prepared data.yaml; identity must be unchanged")
    args = parser.parse_args(argv)
    try:
        run_test(args.selection, local_path(args.output), device=args.device, data=args.data)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

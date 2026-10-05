"""Train on train/val only and export a reproducible handoff for participant B.

Example: python -m src.training.train --config configs/a/smoke.yaml --dry-run
No torch or Ultralytics import occurs until a real training run begins.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {"name", "model", "data", "seed", "train", "augmentation"}
TRAIN_KEYS = {
    "epochs", "imgsz", "batch", "workers", "patience", "optimizer", "lr0", "lrf",
    "weight_decay", "warmup_epochs", "cos_lr", "close_mosaic", "fraction", "amp", "cache",
}
AUGMENTATION_KEYS = {
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup",
}


class ConfigurationError(ValueError):
    """A configuration or prepared dataset is invalid."""


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a mapping")
    return value


def _keys(value: dict, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown or missing:
        raise ConfigurationError(
            f"{label}: unknown keys={sorted(map(str, unknown))}, missing keys={sorted(missing)}"
        )


def _number(value: Any, label: str, low: float, high: float | None = None,
            *, integer: bool = False, exclusive_low: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be numeric")
    if integer and not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer")
    if not math.isfinite(value) or value < low or (exclusive_low and value == low):
        raise ConfigurationError(f"{label} is outside its allowed range")
    if high is not None and value > high:
        raise ConfigurationError(f"{label} must be <= {high}")


def validate_config(config: dict) -> None:
    _keys(config, CONFIG_KEYS, "config")
    name = config["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ConfigurationError("name must contain only letters, digits, underscores or hyphens")
    for key in ("data", "model"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ConfigurationError(f"{key} must be a nonempty path")
    if not config["model"].endswith(".pt"):
        raise ConfigurationError("model must be pretrained .pt weights")
    _number(config["seed"], "seed", 0, 2**32 - 1, integer=True)
    train = _mapping(config["train"], "train")
    aug = _mapping(config["augmentation"], "augmentation")
    _keys(train, TRAIN_KEYS, "train")
    _keys(aug, AUGMENTATION_KEYS, "augmentation")
    for key in ("epochs", "batch", "imgsz"):
        _number(train[key], key, 1, integer=True)
    if train["imgsz"] % 32:
        raise ConfigurationError("imgsz must be divisible by 32")
    for key in ("workers", "patience", "close_mosaic"):
        _number(train[key], key, 0, integer=True)
    for key in ("cos_lr", "amp"):
        if not isinstance(train[key], bool):
            raise ConfigurationError(f"{key} must be true or false")
    if train["cache"] is not False and train["cache"] != "ram":
        raise ConfigurationError("cache must be false or ram")
    if train["optimizer"] not in {"SGD", "Adam", "AdamW"}:
        raise ConfigurationError("optimizer must be SGD, Adam or AdamW")
    for key in ("lr0", "lrf", "fraction"):
        _number(train[key], key, 0, 1, exclusive_low=True)
    _number(train["weight_decay"], "weight_decay", 0, 1)
    _number(train["warmup_epochs"], "warmup_epochs", 0)
    for key, value in aug.items():
        high = {"degrees": 180, "shear": 180, "perspective": 0.001}.get(key, 1)
        _number(value, key, 0, high)


def _absolute(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        try:
            converted = {int(k): v for k, v in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("dataset class IDs must be integers") from exc
        if len(converted) != len(raw) or set(converted) != set(range(len(raw))):
            raise ConfigurationError("dataset class IDs must be contiguous from zero")
        raw = [converted[i] for i in range(len(converted))]
    if not isinstance(raw, list) or not raw or any(not isinstance(v, str) or not v for v in raw):
        raise ConfigurationError("dataset names must be a nonempty list or ID-to-name mapping")
    if len(set(raw)) != len(raw):
        raise ConfigurationError("dataset class names must be unique")
    return raw


def resolve_dataset(data_path: Path) -> tuple[dict, dict, list[str]]:
    """Resolve relative roots against the YAML file, never a global YOLO folder."""
    with data_path.open(encoding="utf-8-sig") as handle:
        raw = _mapping(yaml.safe_load(handle), "dataset YAML")
    allowed = {"path", "train", "val", "test", "names", "nc"}
    if set(raw) - allowed:
        raise ConfigurationError(f"Unsupported dataset keys: {sorted(set(raw) - allowed)}")
    names = _class_names(raw.get("names"))
    if "nc" in raw and raw["nc"] != len(names):
        raise ConfigurationError("dataset nc does not match names")
    if not isinstance(raw.get("path", "."), str):
        raise ConfigurationError("dataset path must be a string")
    data_root = _absolute(raw.get("path", "."), data_path.parent)
    resolved = {"path": str(data_root), "names": dict(enumerate(names)), "nc": len(names)}
    # Deliberately omit test. Participant B performs final held-out evaluation.
    for split in ("train", "val"):
        value = raw.get(split)
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, str) or not v for v in values):
            raise ConfigurationError(f"dataset {split} must be a path or nonempty list of paths")
        paths = [_absolute(v, data_root) for v in values]
        for path in paths:
            if not path.is_dir():
                raise ConfigurationError(f"dataset {split} must use existing image directories: {path}")
            if not any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                       for p in path.rglob("*")):
                raise ConfigurationError(f"dataset {split} has no supported images: {path}")
        resolved[split] = [str(p) for p in paths] if isinstance(value, list) else str(paths[0])
    manifest_path = data_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(f"Missing dataset manifest; run data preparation first: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig") as handle:
        manifest = _mapping(json.load(handle), "dataset manifest")
    categories = manifest.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ConfigurationError("dataset manifest must contain categories")
    if any(not isinstance(c, dict) or not isinstance(c.get("id"), int) or isinstance(c.get("id"), bool)
           for c in categories):
        raise ConfigurationError("dataset manifest categories require integer IDs")
    ordered = sorted(categories, key=lambda item: item["id"])
    if [c["id"] for c in ordered] != list(range(len(names))) or [c.get("name") for c in ordered] != names:
        raise ConfigurationError("dataset manifest classes do not match data.yaml")
    return resolved, manifest, names


def prepare_run(config_path: str | Path, overrides: dict | None = None,
                workspace: Path | None = None) -> dict:
    workspace = (workspace or WORKSPACE).resolve()
    config_path = _absolute(config_path, workspace)
    with config_path.open(encoding="utf-8-sig") as handle:
        config = deepcopy(_mapping(yaml.safe_load(handle), "config"))
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    if set(overrides) - {"data", "name", "device", "epochs", "batch", "imgsz", "fraction", "participant"}:
        raise ConfigurationError("Unsupported command-line override")
    for key in ("data", "name"):
        if key in overrides:
            config[key] = overrides[key]
    for key in ("epochs", "batch", "imgsz", "fraction"):
        if key in overrides:
            _mapping(config.get("train"), "train")[key] = overrides[key]
    validate_config(config)
    device = str(overrides.get("device", "auto"))
    if device not in {"auto", "cpu", "mps"} and not re.fullmatch(r"\d+", device):
        raise ConfigurationError("device must be auto, cpu, mps, or one GPU index (for example 0)")
    data_path = _absolute(config["data"], workspace)
    dataset, manifest, names = resolve_dataset(data_path)
    model = config["model"]
    model_path = (workspace / "models" / "pretrained" / model).resolve() if re.fullmatch(r"yolov8[nslmx]\.pt", model) else _absolute(model, workspace)
    if not model_path.is_file() and not re.fullmatch(r"yolov8[nslmx]\.pt", model):
        raise ConfigurationError(f"Custom pretrained weights do not exist: {model_path}")
    participant = overrides.get("participant", "a")
    if participant not in {"a", "b"}:
        raise ConfigurationError("participant must be a or b")
    run_dir = workspace / "runs" / participant / config["name"]
    handoff_dir = workspace / "models" / participant / config["name"]
    for path in (run_dir, handoff_dir):
        if path.exists():
            raise ConfigurationError(f"Output already exists: {path}. Choose a new --name.")
    return {
        "workspace": workspace, "config_path": config_path, "config": config,
        "device": device, "data_path": data_path, "dataset": dataset,
        "manifest": manifest, "names": names, "model_path": model_path,
        "run_dir": run_dir, "handoff_dir": handoff_dir,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_snapshot(workspace: Path) -> dict:
    versions = {}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "PyYAML", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": versions}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=workspace, capture_output=True, text=True, timeout=10)
        result["git"] = {
            "revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.TimeoutExpired):
        result["git"] = {"revision": None, "dirty": None}
    return result


def _choose_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def _metrics(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", None)
    if not isinstance(values, dict):
        raise RuntimeError("Training completed without validation metrics")
    metrics = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics[str(key)] = number
    if not metrics:
        raise RuntimeError("Training completed without finite validation metrics")
    return metrics


def run_training(plan: dict, *, model_factory: Any = None) -> dict:
    """Reserve output once, train, and export weights plus provenance on success."""
    run_dir, handoff_dir = plan["run_dir"], plan["handoff_dir"]
    if handoff_dir.exists():
        raise ConfigurationError(f"Handoff already exists: {handoff_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "started_at": _now(), "name": plan["config"]["name"]}
    _write_json(run_dir / "status.json", status)
    try:
        config = plan["config"]
        environment = environment_snapshot(plan["workspace"])
        _write_json(run_dir / "environment.json", environment)
        shutil.copy2(plan["config_path"], run_dir / "source_config.yaml")
        shutil.copy2(plan["data_path"], run_dir / "source_data.yaml")
        _write_json(run_dir / "dataset_manifest.json", plan["manifest"])
        _write_json(run_dir / "classes.json", {str(i): value for i, value in enumerate(plan["names"])})
        resolved_path = run_dir / "dataset.resolved.yaml"
        resolved_path.write_text(yaml.safe_dump(plan["dataset"], allow_unicode=True, sort_keys=False), encoding="utf-8")
        # Ultralytics writes settings on import; keep them inside this repository.
        os.environ["YOLO_CONFIG_DIR"] = str(plan["workspace"] / ".cache" / "ultralytics")
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(plan["workspace"] / ".cache" / "matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        device = _choose_device(plan["device"])
        args = {
            **config["train"], **config["augmentation"],
            "data": str(resolved_path), "project": str(run_dir.parent), "name": run_dir.name,
            # Our atomic mkdir already rejected pre-existing runs; let YOLO use this reserved folder.
            "exist_ok": True, "seed": config["seed"], "deterministic": True,
            "device": device, "pretrained": True, "val": True, "split": "val",
            "save": True, "plots": True, "resume": False,
        }
        effective = {**config, "model_resolved": str(plan["model_path"]), "train_arguments": args}
        (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        plan["model_path"].parent.mkdir(parents=True, exist_ok=True)
        model = model_factory(str(plan["model_path"]))
        result = model.train(**args)
        trainer = getattr(model, "trainer", None)
        runtime_amp = getattr(trainer, "amp", args["amp"])
        if hasattr(runtime_amp, "item"):
            runtime_amp = runtime_amp.item()
        effective["runtime"] = {"amp": bool(runtime_amp), "device": device}
        (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
        actual_run = Path(getattr(trainer, "save_dir", run_dir)).resolve()
        if actual_run != run_dir.resolve():
            raise RuntimeError(f"Unexpected output directory: {actual_run}")
        best = run_dir / "weights" / "best.pt"
        if not best.is_file() or best.stat().st_size == 0:
            raise RuntimeError("Training did not produce nonempty weights/best.pt")
        validation = {"split": "val", "metrics": _metrics(result), "test_evaluated": False}
        _write_json(run_dir / "validation_metrics.json", validation)
        provenance = {
            "data_yaml_sha256": sha256(plan["data_path"]),
            "dataset_manifest_sha256": sha256(plan["data_path"].parent / "dataset_manifest.json"),
            "source_config_sha256": sha256(plan["config_path"]),
        }
        pretrained = plan["model_path"]
        if pretrained.is_file():
            provenance["pretrained_weights_sha256"] = sha256(pretrained)
        metadata = {
            "schema_version": 1, "status": "complete", "name": config["name"], "created_at": _now(),
            "task": "detect", "model": config["model"], "weights": "best.pt", "weights_sha256": sha256(best),
            "classes": {str(i): value for i, value in enumerate(plan["names"])},
            "image_size": args["imgsz"], "seed": config["seed"], "device": device,
            "validation": validation, "provenance": provenance, "environment": environment,
            "dataset": plan["manifest"], "effective_config": effective,
            "limitations": plan["manifest"].get("limitations", []),
            "notes": ["Validation was used for training/checkpoint selection; these are not test metrics.",
                      "One-epoch smoke runs only check integration and do not establish model quality."],
        }
        handoff_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(best, handoff_dir / "best.pt")
        for filename in ("classes.json", "dataset_manifest.json", "effective_config.yaml", "environment.json", "validation_metrics.json"):
            shutil.copy2(run_dir / filename, handoff_dir / filename)
        _write_json(handoff_dir / "metadata.json", metadata)
        status.update({"status": "complete", "finished_at": _now(), "handoff": str(handoff_dir)})
        _write_json(run_dir / "status.json", status)
        return metadata
    except BaseException as exc:
        status.update({"status": "failed", "finished_at": _now(), "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(run_dir / "status.json", status)
        if handoff_dir.exists():
            _write_json(handoff_dir / "status.json", status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration relative to repository root")
    parser.add_argument("--data", help="Prepared data.yaml (requires sibling dataset_manifest.json)")
    parser.add_argument("--device", help="auto (default), cpu, mps, or GPU index such as 0")
    parser.add_argument("--participant", choices=("a", "b"), default="a", help="Owner of runs/ and models/ output directories")
    parser.add_argument("--name", help="Unique run name; existing results are never overwritten")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--fraction", type=float, help="Training subset fraction in (0, 1]; record changes when comparing runs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without importing torch/Ultralytics or writing files")
    args = parser.parse_args(argv)
    try:
        plan = prepare_run(args.config, {key: value for key, value in vars(args).items() if key not in {"config", "dry_run"}})
        if args.dry_run:
            print(json.dumps({
                "status": "valid", "config": plan["config"], "dataset": plan["dataset"],
                "device": plan["device"], "run_dir": str(plan["run_dir"]), "handoff_dir": str(plan["handoff_dir"]),
                "limitations": plan["manifest"].get("limitations", []), "test_evaluated": False,
            }, ensure_ascii=False, indent=2))
        else:
            run_training(plan)
            print(f"Training complete. Participant B handoff: {plan['handoff_dir']}")
        return 0
    except (ConfigurationError, OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

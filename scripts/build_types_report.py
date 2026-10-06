"""Build the type-detection result, genuine examples and a portable model archive."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.dataset import Truth, read_json
from src.evaluation.evaluate import load_selection
from src.evaluation.metrics import match
from src.inference.predict import CLASS_LABELS, Detection, Detector, local_path
from src.training.train import sha256


def build(reports: Path, name: str | None = None, archive_name: str | None = None, *, recommend: bool = False) -> None:
    validation = read_json(reports / "validation/comparison.json")
    test = read_json(reports / "test/comparison.json")
    dataset = read_json(ROOT / "reports/multiclass/dataset/SUMMARY.json")
    selection_path = reports / "validation/selection.json"
    selected = load_selection(selection_path)
    name = name or selected["name"]
    if test["selection_sha256"] != sha256(selection_path):
        raise ValueError("Test does not correspond to the current frozen selection")
    if selected["name"] != name or test["models"][0]["name"] != name:
        raise ValueError("Model name does not match the frozen reports")
    # Preserve the actual epoch history alongside the metrics, including attempts
    # that did not win. Image-heavy training outputs stay in ignored runs/.
    for candidate in validation["models"]:
        source_run = ROOT / f"runs/b/{candidate['name']}"
        history = reports / f"training/{candidate['name']}"
        history.mkdir(parents=True, exist_ok=True)
        for filename in ("results.csv", "effective_config.yaml", "source_config.yaml", "status.json"):
            source_file = source_run / filename
            if source_file.is_file():
                shutil.copy2(source_file, history / filename)
    detector = Detector(selected["weights"], str(test["settings"]["device"]))
    if detector.names != {0: "bottle", 1: "bag", 2: "can"}:
        raise ValueError("This report requires the bottle/bag/can checkpoint")
    rows = read_json(reports / f"test/{name}/per_image.json")
    demo = reports / "demo"
    demo.mkdir(exist_ok=False)
    examples = []
    for class_id, class_name in detector.names.items():
        candidates = []
        for row in rows:
            detections = [Detection(d["class_id"], d["class_name"], d["confidence"], tuple(d["xyxy"])) for d in row["detections"]]
            truth = [Truth(t["class_id"], tuple(t["xyxy"])) for t in row["truth"]]
            matched = match(detections, truth, 0.5)
            correct = [detections[i].confidence for i in matched["tp"] if detections[i].class_id == class_id]
            if correct:
                candidates.append((max(correct), row["image"], "correct_detection"))
        if candidates:
            _, relative, status = max(candidates)
        else:
            fallback = [row for row in rows if any(t["class_id"] == class_id for t in row["truth"])]
            if not fallback:
                raise ValueError(f"Test has no ground truth for {class_name}")
            relative, status = fallback[0]["image"], "no_correct_test_detection"
        source = local_path(test["settings"]["data"]).parent / relative
        input_path = demo / f"{class_name}-input{source.suffix}"
        shutil.copy2(source, input_path)
        prediction = detector.predict(source, conf=selected["conf"], imgsz=selected["settings"]["imgsz"],
                                      iou=selected["settings"]["nms_iou"], max_det=selected["settings"]["max_det"])
        prediction.annotated().save(demo / f"{class_name}-result.png")
        (demo / f"{class_name}-detections.json").write_text(json.dumps(prediction.to_dict(), indent=2), encoding="utf-8")
        examples.append({"class": class_name, "source": relative, "status": status,
                         "input": input_path.name, "result": f"{class_name}-result.png"})
    (demo / "index.json").write_text(json.dumps({"examples": examples,
        "note": "Examples illustrate per-class detections after frozen test evaluation; they are not used to tune the model or compute aggregate metrics."}, indent=2), encoding="utf-8")
    with (ROOT / f"runs/b/{name}/results.csv").open(newline="", encoding="utf-8") as handle:
        epochs = len(list(csv.DictReader(handle)))
    metadata = read_json(ROOT / f"models/b/{name}/metadata.json")
    effective = metadata["effective_config"]
    train = effective["train_arguments"]
    validation_row = next(row for row in validation["models"] if row["name"] == name)
    lines = ["# Распознавание типов мусора — результат", "",
             "Модель YOLOv8n различает три типа предметов: **бутылка, пакет, банка**. Для каждого предмета возвращаются рамка, тип и уверенность.", "",
             f"Обучение в этом запуске: {epochs} эпох, {train['imgsz']}×{train['imgsz']}, batch={train['batch']}, seed={effective['seed']}, {train['optimizer']}, {'AMP' if effective['runtime']['amp'] else 'FP32'}. Веса: `models/b/{name}/best.pt`.", "",
             "## Данные", "",
             "Источник — [TACO](https://github.com/pedropro/TACO), официальная проверенная разметка. Это отдельный эксперимент, исходный UAVVaste не получает выдуманные метки типов.", "",
             "| Часть | Фото | Размеченные предметы | Фото без целевых предметов |", "|---|---:|---:|---:|"]
    for split, count in dataset["images"]["by_split"].items():
        lines.append(f"| {split} | {count['images']} | {count['objects']} | {count['negative_images']} |")
    lines += ["", "Категории объединены по исходным меткам TACO: пластиковые/стеклянные бутылки → bottle; бумажные/мусорные/обычные пакеты → bag; пищевые/аэрозольные/напиточные банки → can. Крышки, плёнка и обёртки не считаются пакетами.", "",
              "## Validation и итоговый test", "",
              "| Набор | mAP50 | mAP50–95 | Рабочий Precision | Рабочий Recall | Рабочий F1 |", "|---|---:|---:|---:|---:|---:|"]
    for label, row in (("validation", validation_row), ("test", test["models"][0])):
        metrics = row["metrics"]; op = row["operating_point"]
        lines.append(f"| {label} | {metrics['mAP50']:.4f} | {metrics['mAP50_95']:.4f} | {op['precision']:.4f} | {op['recall']:.4f} | {op['f1']:.4f} |")
    if len(validation["models"]) > 1:
        lines += ["", "## Сравнение до и после на validation", "",
                  "| Модель | mAP50 | mAP50–95 | Рабочий Recall | Рабочий F1 | Порог |",
                  "|---|---:|---:|---:|---:|---:|"]
        for candidate in validation["models"]:
            m, op = candidate["metrics"], candidate["operating_point"]
            lines.append(f"| {candidate['name']} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {op['recall']:.4f} | {op['f1']:.4f} | {op['conf']:.2f} |")
        lines += ["", f"Выбрана **{name}** по максимальному validation mAP50–95. Для каждой модели её порог выбран на validation по micro F1."]
    if not effective["model"].startswith("yolov8"):
        lines += ["", f"Дообучение началось из `{effective['model']}` с lr0={train['lr0']}. Оптимизатор и расписание обучения созданы заново; новые размеченные морские снимки не добавлялись."]
    lines += ["", f"Рабочий порог **{selected['conf']:.2f}** выбран на validation по micro F1 и зафиксирован до test. AP вычисляется отдельно с conf=0.001.", "",
              "## Test по классам", "",
              "| Класс | mAP50 | mAP50–95 | Рабочий Precision | Рабочий Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    row = test["models"][0]
    for class_name, metrics in row["class_metrics"].items():
        op = row["class_operating_points"][class_name]
        lines.append(f"| {CLASS_LABELS[class_name]} | {metrics['mAP50']:.4f} | {metrics['mAP50_95']:.4f} | {op['precision']:.4f} | {op['recall']:.4f} | {op['f1']:.4f} | {op['tp']} | {op['fp']} | {op['fn']} |")
    lines += ["", "## Примеры", "", "Примеры выбраны после итоговой проверки для демонстрации; все TP/FP/FN сохраняются в полном test-отчёте."]
    for example in examples:
        lines += ["", f"### {CLASS_LABELS[example['class']]}", "",
                  f"![{example['class']}](demo/{example['result']})",
                  f"Вход: `demo/{example['input']}`; статус: `{example['status']}`."]
    lines += ["", "## Ограничения", "",
              "Остальные типы мусора модель не классифицирует. Фоновые примеры могут содержать другой мусор и не означают чистую сцену. Пакетов меньше, чем бутылок; метрики нужно смотреть отдельно по классам.", "",
              "Исходные batch-группы TACO сохраняются целиком в одном split, но независимость мест/авторов не подтверждена. Это один seed. Качество на морских, спутниковых и дроновых фото отдельно не измерено; численные результаты относятся к TACO.", "",
              "Сравнивать эти числа напрямую с таблицей UAVVaste нельзя: изменились датасет и классы.", "",
              "Запуск и повторение: [README.md](README.md). Подробные метрики: [validation/REPORT.md](validation/REPORT.md), [test/REPORT.md](test/REPORT.md). История эпох всех сравниваемых моделей сохранена в `training/<имя>/results.csv`."]
    if (ROOT / "reports/multiclass/test/comparison.json").is_file() and reports.resolve() != (ROOT / "reports/multiclass").resolve():
        lines += ["", "Test TACO уже оценивался в первом эксперименте. Здесь это повторная проверка на том же наборе, а не новый независимый test. В этом запуске модель и порог зафиксированы на validation до повторной проверки test."]
    (reports / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    recommendation = ROOT / "reports/multiclass/recommended.json"
    if recommend:
        recommendation.write_text(json.dumps({"selection": selection_path.relative_to(ROOT).as_posix(),
            "model": name, "criterion": selected["criterion"], "note": "Chosen on validation before test evaluation."}, indent=2)+"\n", encoding="utf-8")
    archive = ROOT / f"models/b/{archive_name or name}_handoff.zip"
    folders = [ROOT / f"models/b/{name}", reports]
    reference = local_path(selected["settings"]["reference"])
    if not reference.is_relative_to(reports):
        folders.append(reference)
    for candidate in validation["models"]:
        if "weights" in candidate:
            model_folder = local_path(candidate["weights"]).parent
            if model_folder not in folders:
                folders.append(model_folder)
    parent_model = local_path(effective["model"])
    if parent_model.is_file() and (parent_model.parent / "metadata.json").is_file() and parent_model.parent not in folders:
        folders.append(parent_model.parent)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for folder in folders:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    bundle.write(file, file.relative_to(ROOT).as_posix())
        if recommend:
            bundle.write(recommendation, recommendation.relative_to(ROOT).as_posix())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"model": name, "epochs": epochs, "classes": detector.names,
                      "test_metrics": row["metrics"], "examples": examples, "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/multiclass"))
    parser.add_argument("--name", help="Expected validation winner; omit to use the frozen selection")
    parser.add_argument("--archive-name", help="Unique archive stem for this experiment")
    parser.add_argument("--recommend", action="store_true", help="Use the frozen winner as the application's default")
    args = parser.parse_args()
    build(local_path(args.reports), args.name, args.archive_name, recommend=args.recommend)

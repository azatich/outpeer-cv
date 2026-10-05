"""Create the final B summary from measured results, without retraining or retuning."""

from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build():
    validation = read("reports/b/validation/comparison.json")
    test = read("reports/b/test/comparison.json")
    latency = read("reports/b/latency.json")
    errors = read("reports/b/validation/error_breakdown.json")
    training = read("reports/b/experiments/b_s_baseline_20/metadata.json")
    rows = validation["models"]
    baseline = next(row for row in rows if row["name"] == "a_n_baseline_20")
    larger = next(row for row in rows if row["name"] == "b_s_baseline_20")
    winner = validation["selection"]
    selected = next(row for row in rows if row["name"] == winner["name"])
    final = test["models"][0]
    increase = (larger["metrics"]["mAP50_95"]-baseline["metrics"]["mAP50_95"])*100
    n_time = latency["models"][baseline["name"]]["wall_ms"]["mean_ms"]
    s_time = latency["models"][larger["name"]]["wall_ms"]["mean_ms"]
    lines = ["# Часть Б — результаты", "", "Эксперимент выполнен 5 октября 2026 года.", "",
             f"**YOLOv8s дала прирост validation mAP50–95 на {increase:.2f} процентного пункта** относительно базовой YOLOv8n. Для приложения выбрана `{winner['name']}`, порог **{winner['conf']:.2f}**. Полная обработка подготовленного RGB-кадра в дополнительном замере заняла в среднем {s_time:.1f} мс против {n_time:.1f} мс у базовой YOLOv8n (изменение {(s_time/n_time-1)*100:+.1f}%).", "",
             "## Что выполнено", "",
             "Восстановлены 772 изображения и исходное разбиение А, проверены SHA-256 обеих переданных моделей. YOLOv8s обучена 20 эпох на RTX 3050 Laptop 4 ГБ с параметрами baseline А: imgsz=640, batch=4, seed=42, AdamW, AMP=false, та же аугментация. Обучение заняло около 10 минут. Настройки, логи и история обучения сохранены в [experiments/b_s_baseline_20](experiments/b_s_baseline_20/).", "",
             "Реализованы inference, CLI, единая оценка, подбор порога на validation, фиксация решения перед test, анализ ошибок, Streamlit и два выполненных ноутбука. Часть А интегрирована в ту же ветку; общий модуль обучения сохраняет результаты Б отдельно.", "",
             "## Сравнение на validation", "", "160 изображений / 583 объекта. Значения ниже пересчитаны в едином протоколе: square 640, batch=1, FP32, NMS IoU=0.7, AP conf=0.001. Они могут отличаться от исходного отчёта А с настройками validation внутри обучения.", "",
             "| Модель | Precision¹ | Recall¹ | mAP50 | mAP50–95 | Рабочий порог | F1² |", "|---|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for row in rows:
        metrics, op = row["metrics"], row["operating_point"]
        lines.append(f"| {row['name']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['mAP50']:.4f} | {metrics['mAP50_95']:.4f} | {op['conf']:.2f} | {op['f1']:.4f} |")
        csv_rows.append({"model": row["name"], **metrics, "conf": op["conf"], "operating_f1": op["f1"], "wall_mean_ms": latency["models"][row["name"]]["wall_ms"]["mean_ms"]})
    lines += ["", "¹ P/R Ultralytics относятся к её собственной точке максимального F1. ² F1 рабочего порога: отдельное сопоставление рамок по классу и IoU≥0.5, по убыванию уверенности, один к одному. Порог выбирался по максимуму micro F1 только на validation.", "",
              "Модель выбрана по максимальному mAP50–95; больший размер сети дал улучшение в этом запуске. Аугментированная YOLOv8n уступила baseline. Это один seed и один split, поэтому статистическая значимость преимущества не установлена.", "", "![Качество и скорость](quality_speed.png)", "",
              "## Скорость", "", f"Устройство: **{latency['device_name']}**, PyTorch {training['environment']['packages']['torch']}, FP32, 640×640, batch=1, общий conf=0.25. 20 одинаковых validation-фото × 3 повтора, 3 прогрева на модель. Входные разрешения перечислены в подробном validation-отчёте. GPU синхронизирован; порядок моделей на каждом кадре перемешан с seed=42. В памяти хранится один декодированный RGB-снимок за раз.", "",
              "| Модель | Вся функция, mean мс | median мс | p95 мс | Нейросеть, mean мс | median мс |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        speed = latency["models"][row["name"]]; wall, net = speed["wall_ms"], speed["inference"]
        lines.append(f"| {row['name']} | {wall['mean_ms']:.1f} | {wall['median_ms']:.1f} | {wall['p95_ms']:.1f} | {net['mean_ms']:.1f} | {net['median_ms']:.1f} |")
    lines += ["", "Вся функция включает копирование/нормализацию RGB в API, preprocessing, нейросеть, NMS и получение рамок на CPU. Чтение файла, загрузка модели, рисование и сохранение исключены. Время нейросети отдельно взято из `Results.speed`. У больших исходных фотографий накладные расходы заметны. Есть разброс задержек ноутбука; это измерение этой машины, не универсальный FPS модели.", "",
              "[latency.json](latency.json) содержит все измерения. Первичный последовательный замер внутри [validation/REPORT.md](validation/REPORT.md) дал другую картину wall-time; поэтому выше используется дополнительный прогон с чередованием моделей и меньшим потреблением RAM. Исходный протокол и числа сохранены, решение по качеству не менялось.", "",
              "## Итоговый test", "", "90 изображений / 325 объектов. Проверялась только выбранная модель, с заранее зафиксированным порогом; поиск по test не выполнялся.", "",
              "| Precision¹ | Recall¹ | mAP50 | mAP50–95 |", "|---:|---:|---:|---:|",
              f"| {final['metrics']['precision']:.4f} | {final['metrics']['recall']:.4f} | {final['metrics']['mAP50']:.4f} | {final['metrics']['mAP50_95']:.4f} |", ""]
    op = final["operating_point"]
    lines += [f"При рабочем пороге {op['conf']:.2f}: **TP={op['tp']}, FP={op['fp']}, FN={op['fn']}**, Precision={op['precision']:.4f}, Recall={op['recall']:.4f}, F1={op['f1']:.4f}.", "",
              "Test оказался легче по этим метрикам, чем validation; это не основание перенастраивать модель. Подробности: [test/REPORT.md](test/REPORT.md), [зафиксированное решение](validation/selection.json). Первый технический запуск остановился из-за Windows-доступа к служебному кэшу до расчёта метрик; повтор выполнен с теми же настройками.", "",
              "## Ошибки на validation", "", "У выбранной модели при фиксированном пороге:", "", "| Короткая сторона объекта после resize к 640 | Найдено / всего | Recall |", "|---|---:|---:|"]
    for label, title in [("short_side_lt_8px", "< 8 px"), ("short_side_8_to_16px", "8–16 px"), ("short_side_ge_16px", "≥ 16 px")]:
        values = errors["size_bins"][label]
        lines.append(f"| {title} | {values['found']} / {values['objects']} | {values['recall']:.3f} |")
    fp = errors["false_positive_overlap"]
    lines += ["", f"Очень мелкие объекты пропускаются чаще. Среди FP: {fp['overlap_lt_0.1']} рамки почти не пересекаются с разметкой (IoU<0.1), {fp['overlap_0.1_to_match_iou']} пересекаются недостаточно для зачёта, ещё {fp['duplicate_or_assignment_conflict']} относятся к дубликатам/конфликтам сопоставления. Это геометрические группы, не доказанные причины. Неточная локализация одного объекта может одновременно дать FP и FN."]
    for kind, title in [("tp", "Правильное обнаружение"), ("fn", "Пропуск"), ("fp", "Ложное обнаружение по IoU-критерию")]:
        examples = selected["examples"][kind]
        example = examples[min({"tp": 0, "fn": 1, "fp": 2}[kind], len(examples)-1)]
        lines += ["", f"### {title}", "", f"![{title}](validation/{winner['name']}/{example['file']})", "", f"Источник: `{example['image']}`. Зелёный — TP, красный — FP, жёлтый — рамка пропущенного объекта. Один кадр может содержать разные ошибки."]
    lines += ["", "Полные таблицы и дополнительные кадры: [validation/REPORT.md](validation/REPORT.md), [error_breakdown.json](validation/error_breakdown.json), [ноутбук ошибок](../../notebooks/b/02_error_analysis.ipynb).", "",
              "## Проверка и передача", "", "52 автоматических теста прошли. Проверены реальный inference на GPU, сценарий Streamlit на CPU с настоящим снимком и весами, сброс старого результата при смене порога и HTTP health сервера. Проверка интерфейса выполнена через Streamlit AppTest; браузерная проверка не выполнялась. [Протокол](app_check.json).", "",
              "Локальный архив для передачи: `models/b/participant_b_handoff.zip`, рядом SHA-256. Веса и датасет остаются вне Git. [Инструкция запуска и воспроизведения](README.md).", "",
              "## Ограничения", "", "Группы заданы по именам файлов, независимость полётов и мест не подтверждена. Нет отдельного набора чистых сцен. Результаты не подтверждают качество на морских, прибрежных или спутниковых снимках. Тип материала, площадь загрязнения и уникальное число предметов по видео не определяются. Для дальнейшего улучшения стоит исследовать мелкие объекты и независимые локации в новом эксперименте; текущий test нельзя использовать для подбора настроек."]
    (ROOT / "reports/b/RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (ROOT / "reports/b/comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [row["name"] for row in rows]
    axes[0].bar(names, [row["metrics"]["mAP50_95"] for row in rows], color="#109c80")
    axes[0].set(title="Validation mAP50-95", ylim=(0, 0.5))
    axes[1].bar(names, [latency["models"][name]["wall_ms"]["mean_ms"] for name in names], color="#417fc6")
    axes[1].set(title="Alternating model order; conf=0.25", ylabel="Mean ms / RGB image")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout(); fig.savefig(ROOT / "reports/b/quality_speed.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    build()

"""Build (and optionally execute) the two read-only report notebooks for participant B."""

import argparse
import json
import os
from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(execute=False):
    setup = '''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/b/evaluation.yaml").exists())
validation = ROOT / "reports/b/validation"
report = json.loads((validation / "comparison.json").read_text(encoding="utf-8"))
assert report["split"] == "val" and not report["test_evaluated"]
'''
    notebooks = {
        "01_model_comparison.ipynb": [
            nbf.v4.new_markdown_cell("# Часть Б: YOLOv8n и YOLOv8s\n\nЭтот ноутбук читает выполненные эксперименты, не запускает обучение и не подбирает настройки по test. Разбиение А: 522 train / 160 validation / 90 test."),
            nbf.v4.new_code_cell(setup),
            nbf.v4.new_markdown_cell("## Сопоставимые условия\n\nYOLOv8s повторяет фактический baseline А: 20 эпох, 640, batch=4, seed=42, AdamW, та же аугментация. Скорость всех моделей измеряется заново на одной машине. Один seed не устанавливает статистическую значимость разницы."),
            nbf.v4.new_code_cell('''import yaml
config = yaml.safe_load((ROOT / "configs/b/yolov8s_baseline.yaml").read_text(encoding="utf-8"))
display(pd.Series(config["train"]))
display(pd.Series(report["environment"]["packages"]))
'''),
            nbf.v4.new_markdown_cell("## Качество и скорость на validation\n\nPrecision/Recall в таблице — показатели Ultralytics в её точке максимального F1. Для mAP используется conf=0.001. Рабочий порог выбирается отдельно по micro F1 при IoU=0.5. Время после прогрева включает preprocessing, inference, NMS и получение рамок, но не чтение файла, загрузку весов или рисование."),
            nbf.v4.new_code_cell('''table = pd.DataFrame([{"model": row["name"], **row["metrics"],
    "confidence": row["operating_point"]["conf"], "operating_F1": row["operating_point"]["f1"],
    "initial_sequential_mean_ms": row["latency"]["mean_ms"],
    "device": row["latency"]["device_name"]} for row in report["models"]])
latency = json.loads((ROOT / "reports/b/latency.json").read_text(encoding="utf-8"))
table["alternating_mean_ms"] = table["model"].map(lambda name: latency["models"][name]["wall_ms"]["mean_ms"])
table["network_mean_ms"] = table["model"].map(lambda name: latency["models"][name]["inference"]["mean_ms"])
display(table.round(4))
display(Markdown("Дополнительный замер чередует модели на каждом кадре и хранит один RGB-снимок за раз; общий conf=0.25. Он уменьшает влияние порядка и давления на RAM. Первичный последовательный замер сохранён отдельно."))
display(Image(filename=str(ROOT / "reports/b/quality_speed.png")))
'''),
            nbf.v4.new_code_cell('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
for row in report["models"]:
    curve = pd.DataFrame(row["confidence_curve"])
    ax.plot(curve["conf"], curve["f1"], marker="o", label=row["name"])
ax.set(xlabel="Confidence", ylabel="Validation micro F1", ylim=(0, 1))
ax.legend(); ax.grid(alpha=0.2); plt.show()
'''),
            nbf.v4.new_markdown_cell("## Выбор до итоговой оценки\n\nКритерий выбора модели — максимальный validation mAP50–95. При равном F1 порог выбирается по Precision и затем по большему порогу. Настройки, хэш весов и версия данных фиксируются до test."),
            nbf.v4.new_code_cell('''selection = json.loads((validation / "selection.json").read_text(encoding="utf-8"))
display(pd.Series({k: selection[k] for k in ["name", "conf", "criterion", "weights_sha256", "dataset_identity"]}))
'''),
            nbf.v4.new_markdown_cell("## Итоговый test\n\nНиже только чтение уже выполненного финального отчёта. Эти показатели не используются для выбора другой модели или порога."),
            nbf.v4.new_code_cell('''test = json.loads((ROOT / "reports/b/test/comparison.json").read_text(encoding="utf-8"))
assert test["split"] == "test"
display(pd.DataFrame([{"model": row["name"], **row["metrics"], **{f"operating_{k}": v for k,v in row["operating_point"].items()}} for row in test["models"]]).round(4))
'''),
            nbf.v4.new_markdown_cell("## Ограничения\n\nГруппы получены из имён файлов и не подтверждают независимость мест съёмки. Нет отдельных чистых сцен. Проверка на UAVVaste не устанавливает качество на море, побережье или спутниковых снимках. Подробные выводы: `reports/b/RESULTS.md`."),
        ],
        "02_error_analysis.ipynb": [
            nbf.v4.new_markdown_cell("# Часть Б: разбор ошибок\n\nПоказываются реальные предсказания выбранной модели на validation. Зелёные рамки — TP, красные — FP, жёлтые — пропуски (FN, рамки разметки). Совпадение определяется по классу и IoU≥0.5, одно предсказание на один объект."),
            nbf.v4.new_code_cell(setup),
            nbf.v4.new_code_cell('''winner = report["selection"]["name"]
row = next(row for row in report["models"] if row["name"] == winner)
display(pd.Series(row["operating_point"]))
records = json.loads((validation / winner / "per_image.json").read_text(encoding="utf-8"))
counts = pd.DataFrame([{k: r[k] for k in ["image", "tp", "fp", "fn"]} for r in records])
display(counts.sort_values(["fn", "fp"], ascending=False).head(15))
'''),
            nbf.v4.new_code_cell('''for kind, title in [("tp", "Верные обнаружения"), ("fn", "Пропущенные объекты"), ("fp", "Ложные обнаружения")]:
    display(Markdown("## " + title))
    for example in row["examples"][kind]:
        display(Markdown(example["image"]))
        display(Image(filename=str(validation / winner / example["file"]), width=1000))
    if not row["examples"][kind]:
        display(Markdown("Таких случаев при зафиксированном пороге нет."))
'''),
            nbf.v4.new_markdown_cell("## Как интерпретировать ошибки\n\nПроверьте размер объектов, фон, перекрытие и полноту разметки. Один кадр может содержать все типы ошибок. FP здесь определён относительно имеющихся меток; это не измерение на независимых чистых сценах. Жёлтая рамка не является предсказанием модели. Повышение порога может убрать FP, но увеличивает FN. Неточно локализованная рамка может дать FP и FN одновременно."),
            nbf.v4.new_code_cell('''breakdown = json.loads((validation / "error_breakdown.json").read_text(encoding="utf-8"))
display(pd.DataFrame(breakdown["size_bins"]).T)
display(pd.Series(breakdown["false_positive_overlap"]))
'''),
            nbf.v4.new_code_cell('''display(pd.DataFrame(row["confidence_curve"])[["conf", "tp", "fp", "fn", "precision", "recall", "f1"]].round(4))
'''),
            nbf.v4.new_markdown_cell("## Новая фотография\n\nЗапуск: `python -m streamlit run app/main.py`. Выбранная модель и её порог подставляются автоматически. Изменения в интерфейсе служат демонстрации и не перезаписывают решение эксперимента. Счётчик показывает обнаружения на одном кадре."),
        ],
    }
    folder = ROOT / "notebooks/b"; folder.mkdir(parents=True, exist_ok=True)
    if execute:
        from nbclient import NotebookClient
        kernels = ROOT / ".cache/jupyter/kernels/outpeer-b"
        kernels.mkdir(parents=True, exist_ok=True)
        (kernels / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Outpeer B", "language": "python"}), encoding="utf-8")
        os.environ["JUPYTER_PATH"] = str(ROOT / ".cache/jupyter")
        os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
        os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    for filename, cells in notebooks.items():
        notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        if execute:
            NotebookClient(notebook, timeout=180, kernel_name="outpeer-b", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(notebook, folder / filename)
        print(folder / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    build(parser.parse_args().execute)

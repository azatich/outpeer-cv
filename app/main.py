"""Start from the repository: python -m streamlit run app/main.py."""

from io import BytesIO, StringIO
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.evaluation.evaluate import load_selection
from src.inference.predict import CLASS_LABELS, DEFAULT_WEIGHTS, TYPE_WEIGHTS, Detector, load_image, local_path


@st.cache_resource(show_spinner=False, max_entries=3)
def get_detector(weights: str, device: str, modified_ns: int, size: int) -> Detector:
    return Detector(weights, device)


def available_models() -> dict[str, Path]:
    models = {}
    candidates = [("Типы мусора · осторожное дообучение", ROOT / "models/b/taco_n_types_finetune_careful_30/best.pt"),
                  ("Типы мусора · дообученная", ROOT / "models/b/taco_n_types_finetune_50/best.pt"),
                  ("Типы мусора · бутылка / пакет / банка", TYPE_WEIGHTS),
                  ("YOLOv8n · базовая", DEFAULT_WEIGHTS),
                  ("YOLOv8n · аугментации", ROOT / "models/a/a_n_aug_20/best.pt"),
                  ("YOLOv8s · базовая", ROOT / "models/b/b_s_baseline_20/best.pt")]
    for name, path in candidates:
        if path.is_file():
            models[name] = path
    return models


def render_result(prediction) -> None:
    annotated = prediction.annotated()
    count, confidence, timing = st.columns(3)
    count.metric("Обнаружений на кадре", len(prediction.detections))
    confidence.metric("Порог уверенности", f"{prediction.settings['conf']:.0%}")
    timing.metric("Обработка моделью", f"{prediction.elapsed_ms:.0f} мс")
    by_class = prediction.to_dict()["counts_by_class"]
    if len(prediction.settings.get("classes", {})) > 1:
        for column, (name, total) in zip(st.columns(len(by_class)), by_class.items()):
            column.metric(CLASS_LABELS.get(name, name), total)
    st.image(annotated, caption="Рамки показывают тип предмета и уверенность модели", width="stretch")
    if not prediction.detections:
        st.info("При выбранном пороге объекты выбранных классов не обнаружены. Это не гарантирует, что мусора нет на фотографии.")
    else:
        st.dataframe([{"Класс": CLASS_LABELS.get(d.class_name, d.class_name), "Уверенность": round(d.confidence, 4),
                       "x1": round(d.xyxy[0], 1), "y1": round(d.xyxy[1], 1),
                       "x2": round(d.xyxy[2], 1), "y2": round(d.xyxy[3], 1)} for d in prediction.detections],
                     hide_index=True, width="stretch")
    png = BytesIO(); annotated.save(png, format="PNG")
    csv_buffer = StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    for detection in prediction.detections:
        writer.writerow([detection.class_id, detection.class_name, detection.confidence, *detection.xyxy])
    left, middle, right = st.columns(3)
    left.download_button("Скачать фото", png.getvalue(), "outpeer-result.png", "image/png")
    middle.download_button("Скачать JSON", json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2), "detections.json", "application/json")
    right.download_button("Скачать CSV", csv_buffer.getvalue().encode("utf-8-sig"), "detections.csv", "text/csv")
    st.caption("Время не включает чтение фотографии и рисование рамок; первый анализ может быть медленнее из-за прогрева модели.")


def main() -> None:
    st.set_page_config(page_title="Outpeer · поиск мусора", page_icon="♻️", layout="wide")
    st.title("Поиск мусора на фотографии")
    st.write("Загрузите снимок, запустите анализ и посмотрите найденные объекты.")
    models = available_models()
    selections = {}
    # Read frozen selections before stopping: a trained run can have a custom name.
    selection_files = [ROOT / "reports/b/validation/selection.json"]
    selection_files += sorted((ROOT / "reports/multiclass").rglob("selection.json"))
    recommendation = ROOT / "reports/multiclass/recommended.json"
    if recommendation.is_file():
        try:
            preferred = local_path(json.loads(recommendation.read_text(encoding="utf-8"))["selection"])
            if preferred in selection_files:
                # The last processed valid selection is moved to the first UI option.
                selection_files.remove(preferred)
                selection_files.append(preferred)
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            st.warning("Сохранённый выбор модели недоступен. Выберите модель в настройках.")
    for selection_file in selection_files:
        if not selection_file.is_file():
            continue
        try:
            decision = load_selection(selection_file)
            selected_path = local_path(decision["weights"])
            selections[str(selected_path)] = decision
            if selected_path not in [p.resolve() for p in models.values()]:
                models = {decision["name"]: selected_path, **models}
            for label, path in list(models.items()):
                if path.resolve() == selected_path:
                    selected_label = label if label.endswith(" · выбрана на validation") else f"{label} · выбрана на validation"
                    models = {selected_label: path, **{k: v for k, v in models.items() if k != label}}
                    break
        except FileNotFoundError:
            # A report can refer to a checkpoint that was not transferred.
            # It does not affect the available, installed models.
            continue
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            if models:
                st.warning("Для одной из моделей сохранённые настройки недоступны. Для неё используются ручные настройки.")
    if not models:
        st.warning("Веса модели пока не установлены. Нужен обученный best.pt: инструкция reports/b/README.md или reports/multiclass/README.md.")
        st.stop()
    with st.sidebar:
        st.header("Настройки анализа")
        name = st.selectbox("Модель", list(models))
        weights = models[name]
        selected = selections.get(str(weights.resolve()))
        matches_selection = selected is not None
        initial_conf = float(selected["conf"]) if matches_selection else 0.25
        conf = st.slider("Порог уверенности", 0.01, 1.0, initial_conf, 0.01, key=f"confidence_{name}")
        device = st.selectbox("Устройство", ["auto", "cpu", "0"], format_func=lambda x: {"auto": "Автоматически", "cpu": "Процессор", "0": "Видеокарта NVIDIA"}[x])
        st.caption("Более высокий порог скрывает слабые обнаружения и может увеличить число пропусков.")
        if not matches_selection or abs(conf-initial_conf) > 1e-8:
            st.caption("Ручной порог для демонстрации. Итоговые метрики относятся к настройкам, зафиксированным на validation.")
    settings = selected["settings"] if matches_selection else {"imgsz": 640, "nms_iou": 0.7, "max_det": 300}
    uploaded = st.file_uploader("Фотография", type=["jpg", "jpeg", "png", "webp"], help="Один кадр, до 25 МБ и 40 мегапикселей.")
    if uploaded is None:
        st.session_state.pop("prediction", None)
        st.info("Поддерживаются JPG, PNG и WEBP.")
    else:
        payload = uploaded.getvalue()
        if len(payload) > 25*1024*1024:
            st.error("Файл слишком большой. Максимум — 25 МБ."); st.stop()
        try:
            pixels = load_image(payload, max_pixels=40_000_000)
        except ValueError as exc:
            st.error(f"Не удалось открыть фотографию: {exc}"); st.stop()
        stat = weights.stat()
        signature = (hashlib.sha256(payload).hexdigest(), str(weights), stat.st_mtime_ns, stat.st_size,
                     conf, device, settings["imgsz"], settings["nms_iou"], settings["max_det"])
        if st.session_state.get("prediction_signature") != signature:
            st.session_state.pop("prediction", None)
        with st.expander("Исходная фотография", expanded="prediction" not in st.session_state):
            st.image(pixels, width="stretch")
        if st.button("Запустить анализ", type="primary"):
            try:
                with st.spinner("Ищем мусор на фотографии…"):
                    detector = get_detector(str(weights), device, stat.st_mtime_ns, stat.st_size)
                    st.session_state.prediction = detector.predict(pixels, conf=conf, imgsz=settings["imgsz"], iou=settings["nms_iou"], max_det=settings["max_det"])
                    st.session_state.prediction_signature = signature
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Не удалось выполнить анализ: {exc}")
        if "prediction" in st.session_state:
            render_result(st.session_state.prediction)
    st.divider()
    st.caption("UAVVaste: общий класс rubbish. TACO: bottle (бутылка), bag (пакет), can (банка); остальные типы мусора эта модель не определяет. Счётчик относится к одному кадру. Качество на морских и спутниковых снимках отдельно не проверено.")


if __name__ == "__main__":
    main()

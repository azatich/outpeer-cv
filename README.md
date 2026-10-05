# outpeer-cv

Обнаружение видимого мусора на фотографиях и аэроснимках: подготовка UAVVaste → обучение YOLOv8 → сравнение моделей → приложение Streamlit.

**Часть А:** подготовка данных, анализ и два эксперимента YOLOv8n. **Часть Б (Merey):** обработка новых фото, YOLOv8s, оценка качества и скорости, анализ ошибок и интерфейс.

## Быстрый запуск приложения

Команды PowerShell выполняются из корня проекта, Python 3.13:

```powershell
python -m venv .venv
# NVIDIA; для CPU замените cu124 на cpu:
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-b.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Перед запуском скопируйте из `participant_a_handoff.zip` папки `models/a/a_n_baseline_20/` и `models/a/a_n_aug_20/` в проект. Веса и фотографии исключены из Git. Для первого анализа достаточно `models/a/a_n_baseline_20/best.pt`; полный датасет приложению не нужен.

Откройте `http://localhost:8501`, загрузите фото и нажмите **«Запустить анализ»**. Приложение показывает рамки, уверенность и количество обнаружений; результат можно скачать как PNG, JSON и CSV. Если есть проверенный `reports/b/validation/selection.json`, выбранная на validation модель и её порог используются по умолчанию.

## Структура

| Путь | Назначение |
|---|---|
| `src/data/`, `configs/a/`, `notebooks/a/`, `reports/a/` | Данные и эксперименты участника А |
| `src/training/` | Общий запуск обучения; `--participant b` сохраняет в папки Б |
| `src/inference/` | Фото → рамки, классы, уверенность, изображение и JSON |
| `src/evaluation/` | Проверка split, validation, выбор настроек и отдельный test |
| `app/` | Streamlit |
| `configs/b/`, `notebooks/b/`, `reports/b/` | Настройки, анализ и результаты Б |
| `runs/a/`, `runs/b/`, `models/a/`, `models/b/` | Локальные результаты и веса, без добавления в Git |

## Инструкции и результаты

- [Часть Б: установка, данные, обучение, оценка и приложение](reports/b/README.md)
- [Результаты и выводы Б](reports/b/RESULTS.md)
- [Часть А: инструкция](reports/a/README.md), [результаты](reports/a/RESULTS.md), [передача данных и весов](reports/a/HANDOFF.md)

Проверки: `.\.venv\Scripts\python.exe -m pytest -q`.

Модель распознаёт один класс `rubbish`. Датасет разделён на 522 train / 160 validation / 90 test. Модели и пороги выбираются только на validation; test служит итоговой проверкой выбранной модели. Разбиение по сериям имён не подтверждает независимость мест съёмки. Качество на морских, прибрежных и спутниковых снимках отдельно не установлено. Счётчик относится к одному кадру, а не к уникальным предметам по видео или площади загрязнения.

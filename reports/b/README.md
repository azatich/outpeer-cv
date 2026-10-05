# Часть Б: обработка фотографий, оценка и приложение

Ветка — `merey`. Код А уже входит в исходный `main`. Общий модуль обучения расширен параметром `--participant b`; поведение А по умолчанию сохранено. Рабочие результаты Б находятся в `runs/b/` и `models/b/`.

## Окружение

Установка и запуск приложения — в [общем README](../../README.md). Для NVIDIA используйте CUDA-сборку PyTorch. Проверка:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` остаётся списком зависимостей А; `requirements-b.txt` добавляет Streamlit. `requirements-b-lock.txt` фиксирует полное проверенное окружение Б. Для восстановления CUDA-окружения сначала установите PyTorch по инструкции, затем lock. Для CPU используйте основные два списка, поскольку lock фиксирует CUDA wheels проверенного запуска.

## Восстановление данных и моделей

Из `participant_a_handoff.zip` перенесите **только** `models/a/` и `data/raw/uavvaste/` в соответствующие папки. Не перезаписывайте исходный код поверх текущей ветки. SHA-256 каждой модели указан в её `metadata.json`. Архив передаётся отдельно; в Git весов нет.

```powershell
.\.venv\Scripts\python.exe -m src.data.download
.\.venv\Scripts\python.exe -m src.data.prepare --annotations data/raw/uavvaste/annotations.json --images data/raw/uavvaste/images --groups data/raw/uavvaste/groups_suggested.csv --exif-policy raw-pixels --first-frame --output data/processed/uavvaste
.\.venv\Scripts\python.exe -m src.evaluation.dataset
```

Подготовка требует новой/пустой папки результата. Загрузчик продолжает загрузку большого архива после обрыва: повторите команду. Перед экспериментами проверяются таблицы split и соответствия изображений, источник разметки, классы и политика подготовки относительно `reports/a/dataset/`. При оценке дополнительно проверяются декодированные пиксели выбранного split, наличие и корректность меток. Хэши всех меток фиксируют версию данных между validation и test; содержимое test не используется для выбора настроек.

## Обучение YOLOv8s

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.dataset
.\.venv\Scripts\python.exe -m src.training.train --config configs/b/yolov8s_baseline.yaml --participant b --device 0 --dry-run
.\.venv\Scripts\python.exe -m src.training.train --config configs/b/yolov8s_baseline.yaml --participant b --device 0
```

Параметры повторяют **фактический** baseline А: 20 эпох, 640, batch=4, seed=42, AdamW, те же аугментации, AMP выключен. Меняется архитектура `yolov8n` → `yolov8s`. Pretrained веса загружаются Ultralytics при первом обучении. Аппаратная платформа А и Б различается; сравнение скорости проводится заново на одной машине. Один seed не позволяет установить статистическую значимость разницы.

Готовый checkpoint: `models/b/b_s_baseline_20/best.pt`. Повторные запуски требуют нового `--name`, чтобы не перезаписать результаты; обновите путь в конфигурации оценки. При уменьшении batch/imgsz из-за нехватки памяти это уже изменённый протокол, который нужно явно описать в отчёте.

## Validation, выбор настроек и test

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --config configs/b/evaluation.yaml --output reports/b/validation --device 0
.\.venv\Scripts\python.exe -m src.evaluation.analyze
.\.venv\Scripts\python.exe -m src.evaluation.benchmark
.\.venv\Scripts\python.exe -m src.evaluation.final_test --selection reports/b/validation/selection.json --output reports/b/test --device 0
```

Первая команда оценивает все три модели только на validation. Отсутствующие веса не исключаются из сравнения молча. Последняя команда проверяет **только выбранную модель** на test, без подбора порогов. Она сверяет хэш весов, версию датасета, хэш validation-отчёта и сохранённое решение. Не изменяйте настройки по результатам test и не представляйте повторный подбор как независимую итоговую проверку. Для повторного технического прогона используйте новый `--output`; ранее полученные результаты сохраняются.

- Метрики Ultralytics: Precision, Recall, mAP50 и mAP50–95. Для AP используется `conf=0.001`, чтобы не обрезать PR-кривую порогом интерфейса. Precision/Recall Ultralytics относятся к её собственной оптимальной точке F1.
- Рабочий порог каждой модели выбирается по максимальному micro F1 на validation при IoU=0.5; равенства разрешаются по Precision и затем большему порогу. Рамки сопоставляются по убыванию уверенности, по классу и максимальному IoU, один к одному. Это отдельно подписанные TP/FP/FN и Precision/Recall рабочего порога.
- Модель выбирается по максимальному validation mAP50–95, при равенстве — по имени. Решение сохраняется в `selection.json` до доступа к test.
- Скорость: batch=1, FP32, square resize 640, общий conf=0.25, прогрев, одинаковые 20 кадров и 3 повтора. Измеряется копирование RGB, preprocessing, inference, NMS и получение рамок на CPU; GPU синхронизирован. Чтение файлов, рисование и загрузка модели не входят. Сохраняются все измерения, mean/median/p95, устройство и исходные размеры. Замер проводится без параллельного обучения. Рабочие пороги качества выбираются отдельно и не меняют условия этого замера.
- `REPORT.md`, `comparison.json`, `comparison.png` содержат сравнение. В папках моделей — `summary.json`, `per_image.json`, примеры TP/FP/FN и графики Ultralytics. Зелёный — верное обнаружение, красный — ложное, жёлтый — пропущенная рамка разметки. Один кадр может содержать все три типа.

Автоматизация (train → validation; `-FinalTest` добавляет отдельный финальный шаг):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_b_experiments.ps1 -Device 0 -FinalTest
```

## Анализ отдельной фотографии

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict path/to/photo.jpg --weights models/a/a_n_baseline_20/best.pt --conf 0.25 --device auto --output runs/b/photo_demo
```

Создаются `annotated.png` и `detections.json`. Координаты `xyxy` даны в пикселях фотографии после применения EXIF orientation. Фото преобразуется в RGB; используется первый кадр. `0.25` — демонстрационный порог, а не доказанный оптимум. Выбранный на validation порог указан в отчёте. Выходная папка должна быть новой.

## Интерфейс

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Откройте `http://localhost:8501`. JPG/PNG/WEBP, один кадр до 25 МБ / 40 Мп. Выберите модель, загрузите фото, нажмите «Запустить анализ». Доступны таблица рамок и экспорт PNG/JSON/CSV. Пустой результат показывает ноль обнаружений. При смене фото/модели/порога старый результат сбрасывается. Модель кэшируется; её вызовы защищены блокировкой при одновременной работе нескольких сессий. По умолчанию устройство выбирается автоматически: NVIDIA при наличии, иначе CPU.

Если веса отсутствуют, интерфейс показывает инструкцию, не скачивает другую модель. Если выбранная модель или validation-отчёт изменились, автоматическое решение не используется. Ручной выбор порога в демонстрации не изменяет научные результаты.

## Проверки и ограничения

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты проверяют сопоставление рамок, дубликаты, пустые результаты, повреждённые метки, неизменность выбора перед test, EXIF/RGB, контракт inference, Streamlit и совместимость с кодом А. Реальные веса и GPU для unit-тестов не требуются.

Ноутбуки в `notebooks/b/` читают готовые результаты, объясняют выбор модели и показывают ошибки; повторное обучение/test автоматически из ноутбука не запускаются. После выполнения всех экспериментов сводный отчёт и ноутбуки воспроизводятся командами:

```powershell
.\.venv\Scripts\python.exe scripts/build_b_report.py
.\.venv\Scripts\python.exe scripts/build_b_notebooks.py --execute
```

Скрипт `run_b_experiments.ps1 -FinalTest` выполняет и эти шаги. Итоговая проверка приложения, хэшей и ноутбуков записана в [verification.json](verification.json).

Исходный набор не содержит отдельных чистых сцен. FP относительно текущей разметки не заменяют проверку на чистом побережье; часть ошибок требует проверки полноты разметки. Материалы мусора, площадь в м² и уникальный подсчёт по видео не определяются.

Источники API: [Ultralytics validation](https://docs.ultralytics.com/modes/val/), [Ultralytics predict](https://docs.ultralytics.com/modes/predict/), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [PyTorch 2.6](https://pytorch.org/get-started/previous-versions/#v260).


Дополнительный `latency.json` измеряет те же модели с чередованием порядка на каждом кадре (seed=42), хранит один RGB-снимок за раз и отдельно записывает preprocessing/inference/postprocess из Ultralytics. Это позволяет отличить время нейросети от копирования больших изображений. `error_breakdown.json` содержит recall по размеру объектов и геометрические группы FP. Эти шаги читают validation и не меняют зафиксированный выбор.

Для передачи обученной модели Б подготовлен локальный `models/b/participant_b_handoff.zip` с контрольной суммой `.sha256`. Он содержит папку `models/b/b_s_baseline_20/`; исходники и отчёты передаются через ветку `merey`.

# Дообучение бутылок, пакетов и банок

Этот эксперимент продолжает обучение YOLOv8n из `models/b/taco_n_types_30/best.pt` на прежних 450 train-фотографиях TACO. Размеченные морские фотографии не добавлялись. Исходные веса и первый отчёт сохраняются.

Конфигурация: `configs/multiclass/yolov8n_taco_finetune.yaml`. Максимум 50 дополнительных эпох, lr0=0.0002, AdamW, imgsz=640, batch=4, FP32, seed=42. Early stopping с patience=15; новый оптимизатор и расписание, resume=false. Это не восстановление старого оптимизатора.

Проверяется и осторожный вариант `configs/multiclass/yolov8n_taco_finetune_careful.yaml`: максимум 30 дополнительных эпох от того же исходного best.pt, lr0=0.00005, warmup_epochs=0, batch=8, patience=10. Варианты обучаются независимо; количество их эпох не складывается в одну цепочку весов. Этот вариант выбран по промежуточному поведению validation, без подбора по test.

## Повторение

В проекте должны быть первоначальные веса и подготовленный TACO с прежними пикселями, метками и split. Новый запуск использует отдельные папки:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_types_finetune.ps1 -Device 0 -Name taco_n_types_finetune_v2 -Reports reports/multiclass/finetune_v2
```

Для осторожного варианта с включением первой попытки в сравнение:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_types_finetune.ps1 -Device 0 -Config configs/multiclass/yolov8n_taco_finetune_careful.yaml -Name taco_n_types_finetune_careful_v2 -CompareWith taco_n_types_finetune_50 -Reports reports/multiclass/careful_v2
```

Скрипт дообучает, сравнивает исходную и новую модели на одинаковой validation-части, выбирает по mAP50–95, фиксирует порог по micro F1 и проверяет только победителя на test. Затем проверяет реальное распознавание в Streamlit и тесты, создаёт отчёт и архив. Он отказывается перезаписывать уже существующие результаты.

## Готовый результат

Числа и примеры появятся в [RESULTS.md](RESULTS.md). Подробные отчёты: [validation/REPORT.md](validation/REPORT.md), [test/REPORT.md](test/REPORT.md). `verification.json` фиксирует проверку настоящих весов в Streamlit AppTest; полноценный браузер этим не проверяется.

Приложение читает `reports/multiclass/recommended.json` и ставит проверенного победителя сравнения первым. Веса первого варианта: `models/b/taco_n_types_finetune_50/best.pt` («Типы мусора · дообученная»); осторожного — `models/b/taco_n_types_finetune_careful_30/best.pt` («Типы мусора · осторожное дообучение»). Если исходная модель лучше на validation, она остаётся рекомендованной.

Архив этого сравнения: `models/b/taco_n_types_finetune_careful_30_handoff.zip`; рядом SHA-256. Он содержит веса всех сравниваемых моделей, отчёты, ссылку выбора по умолчанию и эталон split и пикселей. Распакуйте его в корень другого checkout с этим кодом. Обучающие фотографии передаются отдельно; для приложения они не нужны.

## Что означают результаты

Validation используется для выбора весов, модели и порога. Test TACO уже использовался в первом эксперименте: его повторная оценка здесь является сравнением на прежнем наборе, а не новым независимым тестом. Настройки этого запуска не подбираются по test.

Качество на воде, со спутника или дрона отдельно не установлено. Для оценки и дообучения на воде нужен отдельный размеченный набор с разделением по местам или съёмкам.

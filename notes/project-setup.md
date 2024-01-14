# Структура проекта

- Данные, эмбеддинги и модели храним с помощью DVC на s3 - [link](https://console.cloud.yandex.ru/folders/b1gkpsnq6bd5s58dgqre/storage/buckets/sneakers-ml)
- Используем poetry, и pre-commit
- Модели храним в onnx

# Структура проекта

```tree
├── data                  - папка на dvc
│   ├── features          - папка с эмбеддингами картинок
│   ├── merged            - папка с объединёнными датасетами
│   ├── models            - папка с сохранёнными моделями
│   ├── raw               - папка со спаршенными данными
│   └── training          - папка с преобразованными для тренировки данными, сплиты на тренировку и валидацию
├── Dockerfile            - докер для телеграм бота
├── notebooks             - ноутбуки для исследования и первичного написания кода
│   ├── eda
│   ├── features
│   ├── merger
│   ├── models
│   └── parser
├── notes
├── poetry.lock
├── pyproject.toml
├── README.md
├── requirements.txt
└── sneakers_ml
    ├── bot               - скрипты для телеграм бота
    ├── data              - скрипты для парсинга, чистки и объединения данных
    ├── features          - скрипты для генерации эмбеддингов
    ├── models            - скрипты для обучения моделей
    └── utils.py
```

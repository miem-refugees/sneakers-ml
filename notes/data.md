- Data is located on s3 - [link](https://console.cloud.yandex.ru/folders/b1gkpsnq6bd5s58dgqre/storage/buckets/sneakers-ml)
- Перевели все данные на DVC, который лежит на s3.

# Структура проекта

```
.
├── data                                - папка с данными и фичами, хранится на s3 с помощью DVC
│   ├── features                        - эмбеддинги
│   │   └── resnet152.pickle
│   ├── features.dvc
│   ├── merged                          - смердженные данные
│   │   ├── images                      - смердженные картинки
│   │   ├── images.dvc
│   │   ├── metadata                    - метаданные смердженных картинок
│   │   └── metadata.dvc
│   └── raw                             - грязные спаршенные данные
│       ├── images
│       ├── images.dvc
│       ├── metadata
│       └── metadata.dvc
├── LICENSE
├── notebooks                           - ноутбуки
│   ├── eda
│   │   ├── basic-eda.ipynb             - базовое еда
│   │   └── resnet-embedding-eda.ipynb
│   ├── features
│   │   ├── resnet152.ipynb
│   │   └── SIFT.ipynb
│   ├── merger
│   │   ├── collab_splitter.ipynb
│   │   ├── merger.ipynb
│   │   └── rename_path.ipynb
│   └── parser
│       ├── footshop.ipynb
│       ├── highsnobiety_kickscrew.ipynb
│       ├── sneakerbass.ipynb
│       └── superkicks.ipynb
├── notes                               - документация по проекту
│   ├── data.md
│   ├── eda-merging.md
│   ├── img_1.png
│   ├── img_2.png
│   ├── img.png
│   ├── sneakerbaas.md
│   └── superkicks.md
├── poetry.lock
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
├── ├── data                            - модули для парсинга и обработки данных
├── │   ├── base_parser.py              - абстрактный класс для парсинга картинок с сайтов superkicks, sneakerbaas, footshop
├── │   ├── base.py                     - абстрактный класс хранилища
├── │   ├── data_preview.py
├── │   ├── footshop.py                 - парсинг footshop
├── │   ├── image.py                    - функции для работы с картинками
├── │   ├── __init__.py
├── │   ├── local.py                    - локальный класс хранилища, для работы с файлами в локальной системе
├── │   ├── merger.py                   - модуль для предобработки данные и объединения
├── │   ├── __pycache__
├── │   ├── s3.py                       - класс хранилища на s3, для работы с файлами на s3
├── │   ├── sneakerbaas.py              - парсинг sneakerbaas
├── │   ├── storage.py                  - класс для работы с файлами
├── │   ├── superkicks.py               - парсинг superkicks
├── │   └── test
├── ├── features                        - функции для генерации фичей
├── │   ├── __pycache__
├── │   └── resnet152.py                - генерация эмбеддингов resnet152
├── ├── __init__.py
├── ├── models
└── └── __pycache__
    └── └── __init__.cpython-39.pyc

```
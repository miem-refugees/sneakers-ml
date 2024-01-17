# Платформа для поиска похожих кроссовок

## [Цель проекта](https://docs.google.com/document/d/1Gdz3_W7x7L9Ff1-Sl61Cv3L6GHBiceH863Vn1ucXzjU/edit#heading=h.j88xs4dca7be)

Цель данного проекта - построить систему поиска похожих кроссовок по изображениям (задача-CV). В проекте планируется реализовать парсинг данных, а именно картинок и дополнительных метаданных. Далее следует этап чистки, обработки и объединения данных. ML часть проекта будет заключатся в обучении классификаторов изображений кроссовок по брендам. В DL части будет улучшено качество классификации с помощью продвинутных моделей, а так же решены другие задачи, такие как image2image поиск и similarity learning. В результате полученные модели будут обернуты в телеграм бот или streamlit сервис.

## Документация проекта

- [Структура проекта](notes/project-setup.md)
- [Описание данных sneakerbaas](notes/sneakerbaas.md)
- [Описание данных superkicks](notes/superkicks.md)
- [Объединение данных и eda](notes/eda-merging.md)
- [Описание моделей и эмбеддингов](notes/features-models.md)

## Roadmap

- [x] **Поиск и сбор данных**
  - [x] Парсинг [sneakerbaas](https://www.sneakerbaas.com)
  - [x] Парсинг [superkicks](https://www.superkicks.in)
  - [x] Парсинг [highsnobiety](https://www.highsnobiety.com)
  - [x] Парсинг [kickscrew](https://www.kickscrew.com/)
  - [x] Парсинг [footshop](https://www.footshop.com)
  - [x] Выгрузка данных на s3 с помощью DVC
  - [x] Очистка данных
  - [x] Объединение данных в один датасет, готовый для тренировки моделей
  - [x] Документация и описание данных
- [x] **Настройка проекта**
  - [x] Настроить poetry
  - [x] Добавление линтеров и форматеров
  - [x] Добавление pre-commit
- [x] **Получение эмбеддингов**
  - [x] SIFT
  - [x] HOG
  - [x] ResNet152
  - [ ] Сохранение в npy формате
- [x] **Классификация по брендам**
  - [x] Модели классического машинного обучения на полученных эмбеддингах
    - [x] SVM
    - [x] SGD
    - [x] CatBoost
    - [x] Сохранение в onnx формате
  - [ ] Модели глубинного обучения
    - [ ] ResNet
    - [ ] Vision transformer
- [ ] **Обёртка моделей**
  - [x] Telegram bot
  - [ ] streamlit
- [ ] image2image
  - [ ] faiss
- [ ] similarity learning
- [ ] Возможно text2image, image2text

## Пример работы телеграм бота

![ezgif-3-a70e75c32f](https://github.com/miem-refugees/sneakers-ml/assets/57370975/0ded53d5-479d-458a-b1ed-3675b3e1f71c)

## Список членов команды

- Литвинов Вячеслав
- Моисеев Даниил

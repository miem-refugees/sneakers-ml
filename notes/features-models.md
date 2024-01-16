# Data splits

- Работаем с классификацией по брендам
- У некоторых брендов мало картинок.
- Отобрали только те бренды, у которых больше 100 картинок.
- Получилось 13 брендов для классификации.
- Разделили данные на train/val/test в пропорциях 60/20/20.

# Features and models

| feature-model | baseline (most-frequent) | hog-svm | hog-sgd | hog-catboost | resnet-svm | resnet-sgd | resnet-catboost |
| ------------- | ------------------------ | ------- | ------- | ------------ | ---------- | ---------- | --------------- |
| f1-weighted   | 0.12                     | 0.73    | 0.70    | 0.68         | 0.70       | 0.71       | 0.65            |
| f1-macro      | 0.03                     | 0.70    | 0.67    | 0.63         | 0.68       | 0.69       | 0.61            |
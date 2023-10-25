# [superkicks](https://www.superkicks.in/)

## 8 категорий

- men-basketball-sneakers
- men-classics-sneakers
- men-skateboard-sneakers
- men-sneakers
- women-basketball-sneakers
- women-classics-sneakers
- women-skateboard-sneakers
- women-sneakers

## Фотографии

Фотографии хранятся на s3 рассортированы по папкам-брендам. В основном каждая модель имеет несколько фотографий с различных углов:

| Вид | Картинка |
|---|---|
| Справа |  |
| Спереди |  |
| Сзади |  |
| Задняя часть |  |

Есть, модели в которых не хватает фотографий, есть те, где есть дополнительные.

## Metadata

В каждой категогии и в общей папке содержится файл `metadata.csv` с полями:

- **brand** - бренд
- **title** - название модели
- **price** - цена кроссовок
- **manufacturer** - производитель
- **country_of_origin** - страна-производитель
- **imported_by** - компания импортёр
- **weight** - вес кроссовок
- **generic_name**
- **unit_of_measurement** - количество пар
- **marketed_by** - название продавца
- **article_code** - код модели
- **description** - описание
- **collection_name** - название коллекции
- **collection_url** - ссылка на коллекцию
- **url** - ссылка на модель
- **images_dir** - путь к локальной директории с картинками модели
- **s3_dir** - путь к директории картинок модели на s3
- **product_dimensions**

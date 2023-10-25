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
| Справа | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/c4347ddf-cb57-4907-8ca8-26b1228cdcce) |
| Спереди | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/953a8a7b-10c2-4bd1-8011-5ab76def0aa3) |
| Сзади | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/f385b140-c14a-4088-ba55-a2f88a6ded83) |
| Передняя часть | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/f5518b95-6fb1-4b7b-8ebc-01fae3812fb7) |
| Задняя часть | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/3e4750d7-98cd-4ee0-8966-18c37fd6df42) |

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

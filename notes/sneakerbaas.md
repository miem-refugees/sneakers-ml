# [sneakerbaas](https://www.sneakerbaas.com)

## [4 категории](https://www.sneakerbaas.com/collections/sneakers)

- kids
- men
- unisex
- women

## Фотографии

Фотографии хранятся на s3 рассортированы по папкам-брендам. В основном каждая модель имеет по 3 фотографии на белом фоне:

| Вид | Картинка |
|---|---|
| Слева | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/7e728431-1238-4589-9563-9b9dd4d36960) |
| Справа | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/a590de7f-e0f2-47e0-825c-2b4ff788a2e4) |
| Подошва | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/22d2e9e8-df51-4c3b-89b9-c21b0f15ae2e) |

Есть, модели в которых не хватает фотографий, есть те, где есть дополнительные.

## Metadata

В каждой категогии и в общей папке содержится файл `metadata.csv` с полями:

- **brand** - бренд
- **description** - описание, из которого можно достадь дополнительную информацию (цвет модели и т.д.)
- **pricecurrency** - валюта, в которой продаются кроссовки
- **price** - цена кроссовок
- **title** - название модели
- **collection_name** - название коллекции
- **collection_url** - ссылка на коллекцию
- **url** - ссылка на модель
- **images_dir** - путь к локальной директории с картинками модели
- **s3_dir** - путь к директории картинок модели на s3

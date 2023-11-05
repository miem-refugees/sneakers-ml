# Data is located on s3

[Link](https://console.cloud.yandex.ru/folders/b1gkpsnq6bd5s58dgqre/storage/buckets/sneakers-ml)

```
data
├── raw                              <- Raw parsed data from various sources
│   ├── sneakerbaas
│   ├── footshop
│   ├── highsnobiety
│   │   ├── adidas
│   │   │   └── some adidas model
│   │   │       ├── 1.jpg
│   │   │       ├── 2.jpg
│   │   │       ├── 3.jpg
│   │   │       └── ...
│   │   └── ...
│   └── superkicks
│       ├── adidas
│       │   └── some adidas model
│       │       ├── 1.jpg
│       │       ├── 2.jpg
│       │       ├── 3.jpg
│       │       └── ...
│       └── ...   
└── union                            <- Unioned data
    └── adidas
        └── some adidas model
            ├── 1.jpg
            ├── 2.jpg
            ├── 3.jpg
            ├── 4.jpg
            ├── 5.jpg
            ├── 6.jpg
            └── ...
```

```
data
  raw
    sneakerbaas
    footshop
    highsnobiety
      adidas
        some adidas model
          1.jpg
          2.jpg
          3.jpg
          ...
      ...
    superkicks
      adidas
        some adidas model
          1.jpg
          2.jpg
          3.jpg
          ...
      ...   
  union
    adidas
      some adidas model
        1.jpg
        2.jpg
        3.jpg
        4.jpg
        5.jpg
        6.jpg
        ...
```

```shell
├── data
│   ├── raw            
│   ├── raw
│   ├── union           
│   ├── intermediate   <- Data with removed duplicates?
│   ├── processed      <- Data for model training, maybe with resized photos?
```

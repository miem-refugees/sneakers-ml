from pathlib import Path

import pandas as pd
import splitfolders

from sneakers_ml.data.local import LocalStorage
from sneakers_ml.data.storage import StorageProcessor


def move_top_sneakers(dataset_path: str, move_path: str, min_count: int = 100) -> None:
    dataset = pd.read_csv(dataset_path)
    processor = StorageProcessor(LocalStorage())
    imgs = dataset[dataset["unique_images_count"] > min_count]["images"].to_list()
    brands = dataset[dataset["unique_images_count"] > min_count]["brand_merge"].to_list()
    for image_folder, brand in zip(imgs, brands):
        processor.images_to_directory([image_folder], str(Path(move_path) / brand))


def split_train_test_val(input_folder: str, output_folder: str) -> None:
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.6, 0.2, 0.2))


if __name__ == "__main__":
    move_top_sneakers("data/merged/metadata/brands_dataset.csv", "data/training/brands-classification-full", 100)
    split_train_test_val("data/training/brands-classification-full", "data/training/brands-classification-splits")

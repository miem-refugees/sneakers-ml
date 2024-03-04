from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import splitfolders
from hydra import compose, initialize
from omegaconf import DictConfig
from umap import UMAP

from sneakers_ml.data.storage.local import LocalStorage
from sneakers_ml.data.storage.storage import StorageProcessor
from sneakers_ml.features.resnet152 import ResNet152Features


def create_dataframe(
    features_: np.ndarray, classes_: np.ndarray, idx_to_class: dict[int, str], images_: np.ndarray
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        np.concatenate([features_, classes_.reshape(-1, 1), images_.reshape(-1, 1)], axis=1),
        columns=["feature_1", "feature_2", "class", "image"],
    )
    dataframe["feature_1"] = pd.to_numeric(dataframe["feature_1"])
    dataframe["feature_2"] = pd.to_numeric(dataframe["feature_2"])
    dataframe["class_name"] = dataframe["class"].apply(lambda x: idx_to_class[int(x)])
    return dataframe


def filter_images(cfg: DictConfig) -> list[str]:
    features_class = ResNet152Features(cfg.features.resnet152.config, cfg.data)
    features, classes, class_to_idx = features_class.get_features_folder(cfg.data.splits.full)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes_idx = classes[:, 1]
    images_idx = classes[:, 0]

    umap_model = UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(features)

    dataframe = create_dataframe(umap_embeddings, classes_idx, idx_to_class, images_idx)
    plt.scatter(dataframe["feature_1"], dataframe["feature_2"])
    plt.savefig("umap_resnet_full.jpg")

    feature_1_right_boundary = float(input("Enter right boundary for feature_1: "))

    return dataframe[dataframe["feature_1"] <= feature_1_right_boundary]["image"].to_list()


def remove_bad_images(bad_imgs_paths: list[str]) -> None:
    for img in bad_imgs_paths:
        path = Path(img)
        if path.exists():
            path.unlink()


def move_top_sneakers(dataset_path: str, move_path: str, min_count: int = 100) -> None:
    dataset = pd.read_csv(dataset_path)
    processor = StorageProcessor(LocalStorage())
    top_sneakers_dataset = dataset[dataset["unique_images_count"] > min_count]
    imgs = top_sneakers_dataset["images"].to_list()
    brands = top_sneakers_dataset["brand_merge"].to_list()
    for image_folder, brand in zip(imgs, brands):
        processor.images_to_directory([image_folder], str(Path(move_path) / brand))


def split_train_test_val(input_folder: str, output_folder: str) -> None:
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.6, 0.2, 0.2))


def create_brands_classification(cfg: DictConfig) -> None:
    move_top_sneakers(cfg.paths.merged.metadata.brands_dataset, cfg.data.splits.full, 100)
    split_train_test_val(cfg.data.splits.full, cfg.data.path)


def create_brands_classification_filtered(cfg: DictConfig) -> None:
    move_top_sneakers(cfg.paths.merged.metadata.brands_dataset, cfg.data.splits.full, 100)
    bad_imgs_paths = filter_images(cfg)
    remove_bad_images(bad_imgs_paths)
    split_train_test_val(cfg.data.splits.full, cfg.data.path)


if __name__ == "__main__":

    with initialize(version_base=None, config_path="../../config", job_name="create_brands_classification_main"):
        config = compose(config_name="config", overrides=["data=brands_classification"])
        create_brands_classification(config)

    with initialize(version_base=None, config_path="../../config", job_name="create_brands_classification_filtered"):
        config = compose(config_name="config", overrides=["data=brands_classification_filtered"])
        create_brands_classification_filtered(config)

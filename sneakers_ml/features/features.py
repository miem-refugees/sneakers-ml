import pickle
from pathlib import Path
from typing import Callable

import numpy as np


def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as save_file:
        pickle.dump((numpy_features, classes, class_to_idx), save_file)


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def load_split(features_folder_path: str, features_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_path = str(Path(features_folder_path) / f"{features_name}-{split}.pickle")
    numpy_features, classes, _ = load_features(split_path)
    x = numpy_features
    y = classes[:, 1]
    return x, y


def get_train_val_test(
    features_folder_path: str, features_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = load_split(features_folder_path, features_name, "train")
    x_val, y_val = load_split(features_folder_path, features_name, "val")
    x_test, y_test = load_split(features_folder_path, features_name, "test")

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_features(
    images_dataset_path: str,
    save_folder: str,
    feature_creation_func: Callable[[str], tuple[np.ndarray, np.ndarray, dict[str, int]]],
    features_name: str,
) -> None:
    subdirectories = [x.name for x in Path(images_dataset_path).iterdir() if x.is_dir()]
    has_splits = bool(len(set(subdirectories).intersection({"train", "test", "val"})))
    if has_splits:
        for folder in ["train", "test", "val"]:
            split_path = str(Path(images_dataset_path) / folder)

            features, classes, class_to_idx = feature_creation_func(split_path)
            save_path = str(Path(save_folder) / f"{features_name}-{folder}.pickle")
            save_features(save_path, features, classes, class_to_idx)
    else:
        features, classes, class_to_idx = feature_creation_func(images_dataset_path)
        save_path = str(Path(save_folder) / f"{features_name}.pickle")
        save_features(save_path, features, classes, class_to_idx)

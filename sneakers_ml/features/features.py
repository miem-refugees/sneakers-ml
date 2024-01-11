import pickle
from pathlib import Path

import numpy as np


def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as save_file:
        pickle.dump((numpy_features, classes, class_to_idx), save_file)


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def load_split(folder_path: str, dataset_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_path = str(Path(folder_path) / f"{dataset_name}-splits-{split}.pickle")
    numpy_features, classes, _ = load_features(split_path)
    x = numpy_features
    y = classes[:, 1]
    return x, y


def get_train_val_test(
    folder_path: str, dataset_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = load_split(folder_path, dataset_name, "train")
    x_val, y_val = load_split(folder_path, dataset_name, "val")
    x_test, y_test = load_split(folder_path, dataset_name, "test")

    return x_train, x_val, x_test, y_train, y_val, y_test

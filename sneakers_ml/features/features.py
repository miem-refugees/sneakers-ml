import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image


class BaseFeatures(ABC):
    def __init__(self, cfg_data: DictConfig, cfg_features: DictConfig) -> None:
        super().__init__()

        self.cfg_data = cfg_data
        self.cfg_features = cfg_features

    @staticmethod
    def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            pickle.dump((numpy_features, classes, class_to_idx), save_file)

    @staticmethod
    def load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        with Path(path).open("rb") as file:
            return pickle.load(file)

    @classmethod
    def load_split(cls, split_path: str) -> tuple[np.ndarray, np.ndarray]:
        numpy_features, classes, _ = cls.load_features(split_path)
        x = numpy_features
        y = classes[:, 1]
        return x, y

    def load_train_val_test_splits(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, y_train = self.load_split(self.cfg_features.train)
        x_val, y_val = self.load_split(self.cfg_features.val)
        x_test, y_test = self.load_split(self.cfg_features.test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def load_full_split(self) -> tuple[np.ndarray, np.ndarray]:
        return self.load_split(self.cfg_features.full)

    @abstractmethod
    def apply_transforms(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def get_feature(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_features(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        raise NotImplementedError

    def create_features(self) -> None:
        for split in self.cfg_data:
            numpy_features, classes, class_to_idx = self.get_features(self.cfg_data[split])
            self.save_features(self.cfg_features[split], numpy_features, classes, class_to_idx)

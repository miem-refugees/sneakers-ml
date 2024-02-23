import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image


class BaseFeatures(ABC):
    def __init__(self, cfg_features: DictConfig, cfg_data: DictConfig) -> None:
        super().__init__()

        self.cfg_features = cfg_features
        self.cfg_data = cfg_data

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

    def create_features(self) -> None:
        for split in self.cfg_data:
            numpy_features, classes, class_to_idx = self.get_features(self.cfg_data[split])
            self.save_features(self.cfg_features[split], numpy_features, classes, class_to_idx)

    @abstractmethod
    def apply_transforms(self, image: Image.Image) -> Union[Image.Image, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_feature(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_features(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        raise NotImplementedError

    @staticmethod
    def get_device(device: str) -> str:
        if device.lower().startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    @staticmethod
    def crop_image(image: Image.Image, crop_sides: int, crop_top_bot: int) -> Image.Image:
        width, height = image.size
        left = (width - crop_sides) // 2
        top = (height - crop_top_bot) // 2
        right = (width + crop_sides) // 2
        bottom = (height + crop_top_bot) // 2

        return image.crop((left, top, right, bottom))

    @staticmethod
    def to_numpy(tens: torch.Tensor) -> np.ndarray:
        return tens.detach().cpu().numpy() if tens.requires_grad else tens.cpu().numpy()  # type: ignore[no-any-return]

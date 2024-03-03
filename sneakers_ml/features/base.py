from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


class BaseFeatures(ABC):
    def __init__(self, config: DictConfig, config_data: DictConfig) -> None:
        super().__init__()

        self.config = config
        self.config_data = config_data

    @staticmethod
    def _save_features(
        path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]
    ) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            np.save(save_file, numpy_features, allow_pickle=False)
            np.save(save_file, classes, allow_pickle=False)
            np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    @staticmethod
    def _load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        with Path(path).open("rb") as file:
            numpy_features = np.load(file, allow_pickle=False)
            classes = np.load(file, allow_pickle=False)
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            class_to_idx = dict(zip(class_to_idx_numpy[:, 0], class_to_idx_numpy[:, 1].astype(int)))
            return numpy_features, classes, class_to_idx

    @classmethod
    def load_split(cls, split_path: str) -> tuple[np.ndarray, np.ndarray]:
        numpy_features, classes, _ = cls._load_features(split_path)
        x = numpy_features
        y = classes[:, 1]
        return x, y

    def load_train_val_test_splits(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, y_train = self.load_split(self.config.splits.train)
        x_val, y_val = self.load_split(self.config.splits.val)
        x_test, y_test = self.load_split(self.config.splits.test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def load_full_split(self) -> tuple[np.ndarray, np.ndarray]:
        return self.load_split(self.config.splits.full)

    def create_features(self) -> None:
        for split in self.config_data.splits:
            numpy_features, classes, class_to_idx = self.get_features_folder(self.config_data.splits[split])
            self._save_features(self.config.splits[split], numpy_features, classes, class_to_idx)

    def get_class_to_idx(self) -> dict[str, int]:
        class_to_idx_all = [self._load_features(self.config.splits[split])[2] for split in self.config.splits]
        if all(x == class_to_idx_all[0] for x in class_to_idx_all):
            return class_to_idx_all[0]
        msg = "Different class_to_idx in splits"
        raise ValueError(msg)

    @abstractmethod
    def apply_transforms(self, image: Image.Image) -> Union[Image.Image, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_feature(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        raise NotImplementedError

    @staticmethod
    def crop_image(image: Image.Image, crop_sides: int, crop_top_bot: int) -> Image.Image:
        width, height = image.size
        left = (width - crop_sides) // 2
        top = (height - crop_top_bot) // 2
        right = (width + crop_sides) // 2
        bottom = (height + crop_top_bot) // 2

        return image.crop((left, top, right, bottom))


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def create_all_features(cfg: DictConfig) -> None:
    for feature in cfg.features:
        tqdm.write(f"Creating {feature}")
        instantiate(config=cfg.features[feature], config_data=cfg.data).create_features()


if __name__ == "__main__":
    create_all_features()  # pylint: disable=no-value-for-parameter

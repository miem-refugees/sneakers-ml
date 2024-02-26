from collections.abc import Sequence

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from skimage.feature import hog  # pylint: disable=no-name-in-module
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from sneakers_ml.features.base import BaseFeatures


class HogFeatures(BaseFeatures):

    def apply_transforms(self, image: Image.Image) -> Image.Image:
        image_resized = image.resize((256, 256))
        return self.crop_image(image_resized, 224, 224)

    def get_feature(self, image: Image.Image) -> np.ndarray:
        transformed_image = self.apply_transforms(image)

        return hog(  # type: ignore[no-any-return]
            transformed_image,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=False,
            channel_axis=-1,
            feature_vector=True,
        )

    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        features = [self.get_feature(image) for image in images]
        return np.array(features)

    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        dataset = ImageFolder(folder_path)

        features = []
        for image, _ in tqdm(dataset, desc=folder_path):
            feature = self.get_feature(image)
            features.append(feature)

        classes = np.array(dataset.imgs)
        class_to_idx = dataset.class_to_idx
        numpy_features = np.array(features)

        return numpy_features, classes, class_to_idx


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def create_features(cfg: DictConfig) -> None:
    HogFeatures(cfg.features.hog.config, cfg.data).create_features()


if __name__ == "__main__":
    create_features()  # pylint: disable=no-value-for-parameter

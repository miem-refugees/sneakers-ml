import abc

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


class FeatureExtractor(abc.ABC):
    def __init__(self, name):
        self.name = name

    def get_features(self, folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        dataset = ImageFolder(folder)
        features = [self._compute_features(image) for image, _ in tqdm(dataset)]

        classes = np.array(dataset.imgs)
        class_to_idx = dataset.class_to_idx
        numpy_features = features

        return numpy_features, classes, class_to_idx

    @abc.abstractmethod
    def _compute_features(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _crop_image(image: Image.Image, crop_sides: int, crop_top_bot: int) -> Image.Image:
        width, height = image.size
        left = (width - crop_sides) // 2
        top = (height - crop_top_bot) // 2
        right = (width + crop_sides) // 2
        bottom = (height + crop_top_bot) // 2

        return image.crop((left, top, right, bottom))

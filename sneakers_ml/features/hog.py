import numpy as np
from PIL import Image
from skimage.feature import hog
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from sneakers_ml.features import crop_image
from sneakers_ml.features.features import BaseFeatures


class HogFeatures(BaseFeatures):
    def apply_transforms(self, image: Image.Image) -> Image.Image:
        image_resized = image.resize((256, 256))
        return crop_image(image_resized, 224, 224)

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

    def get_features(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        dataset = ImageFolder(folder_path)

        features = []
        for image, _ in tqdm(dataset):
            feature = self.get_feature(image)
            features.append(feature)

        classes = np.array(dataset.imgs)
        class_to_idx = dataset.class_to_idx
        numpy_features = np.array(features)

        return numpy_features, classes, class_to_idx

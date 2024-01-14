import numpy as np
from PIL import Image
from skimage.feature import hog

from sneakers_ml.features.extractor import FeatureExtractor


class HogFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("HOG")

    def _compute_features(self, image: Image.Image) -> np.ndarray:
        return self.get_hog(image)

    def get_hog(self, image: Image.Image) -> np.ndarray:
        image_resized = image.resize((256, 256))
        image_cropped = self._crop_image(image_resized, 224, 128)

        feature = hog(
            image_cropped,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=False,
            channel_axis=-1,
            feature_vector=True,
            transform_sqrt=True,
        )
        return feature

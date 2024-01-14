import cv2
import numpy as np
from PIL import Image

from sneakers_ml.features.extractor import FeatureExtractor


class SiftFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("SIFT")

    def _compute_features(self, image: Image.Image) -> np.ndarray:
        return self.get_sift(image)

    def get_sift(self, image: Image.Image) -> np.ndarray:
        image_resized = image.resize((256, 256))
        image_cropped = self._crop_image(image_resized, 224, 128)
        _, descriptor = cv2.SIFT_create().detectAndCompute(np.asarray(image_cropped), None)
        return descriptor

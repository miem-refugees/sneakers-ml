from collections.abc import Sequence

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from sneakers_ml.features.base import BaseFeatures
from sneakers_ml.models.onnx_utils import get_session, predict, save_sklearn_model


class SIFTFeatures(BaseFeatures):

    def __init__(self, config: DictConfig, config_data: DictConfig) -> None:
        super().__init__(config, config_data)
        self.sift_instance = cv2.SIFT_create()  # type: ignore[attr-defined] # pylint: disable=no-member
        self.kmeans = KMeans(n_clusters=self.config.kmeans.n_clusters, n_init="auto")

        if self.config.kmeans.use_onnx is True:
            self.onnx_session = get_session(self.config.kmeans.onnx_path)

    def apply_transforms(self, image: Image.Image) -> Image.Image:
        image_resized = image.resize((256, 256))
        return self.crop_image(image_resized, 224, 224)

    # methods for creating features and training
    def _get_sift(self, image: Image.Image) -> np.ndarray:
        transformed_image = self.apply_transforms(image)
        _, descriptors = self.sift_instance.detectAndCompute(np.array(transformed_image), None)
        return descriptors  # type: ignore[no-any-return]

    def _get_sift_descriptors(self, folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[np.ndarray]]:
        dataset = ImageFolder(folder)

        images_descriptors = []
        flattened_sift_descriptors = []
        for image, _ in tqdm(dataset, desc=folder):
            descriptors = self._get_sift(image)
            images_descriptors.append(descriptors)

        classes = np.array(dataset.imgs)
        class_to_idx = dataset.class_to_idx

        for image_descriptors in tqdm(images_descriptors):
            if image_descriptors is not None:
                for descriptor in image_descriptors:
                    flattened_sift_descriptors.append(descriptor)  # noqa: PERF402

        return np.array(flattened_sift_descriptors), classes, class_to_idx, images_descriptors

    def _get_feature_vector(self, sift_image_descriptors: np.ndarray) -> np.ndarray:
        image_feature_vector = [0] * self.config.kmeans.n_clusters
        if sift_image_descriptors is not None:
            descriptors_clusters = self.kmeans.predict(sift_image_descriptors)
            for cluster in descriptors_clusters:
                image_feature_vector[cluster] += 1
        return np.array(image_feature_vector)

    def get_feature(self, image: Image.Image) -> np.ndarray:
        descriptors = self._get_sift(image)

        image_feature_vector = [0] * self.config.kmeans.n_clusters
        if descriptors is not None:
            if self.config.kmeans.use_onnx is True:
                clusters = predict(self.onnx_session, descriptors)
            else:
                clusters = self.kmeans.predict(descriptors)
            for cluster in clusters:
                image_feature_vector[cluster] += 1

        return np.array(image_feature_vector)

    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        features = [self.get_feature(image) for image in images]
        return np.array(features)

    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        flattened_sift_descriptors, classes, class_to_idx, images_descriptors = self._get_sift_descriptors(folder_path)

        if not hasattr(self.kmeans, "cluster_centers_"):  # check if kmeans is not fitted
            self.kmeans.fit(flattened_sift_descriptors)
            save_sklearn_model(self.kmeans, flattened_sift_descriptors[:1], self.config.kmeans.onnx_path)

        features = [self._get_feature_vector(images_descriptors) for images_descriptors in tqdm(images_descriptors)]
        return np.array(features), classes, class_to_idx


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def create_features(cfg: DictConfig) -> None:
    SIFTFeatures(cfg.features.sift.config, cfg.data).create_features()


if __name__ == "__main__":
    create_features()  # pylint: disable=no-value-for-parameter

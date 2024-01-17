from typing import Union

import cv2
import numpy as np
import onnxruntime as rt
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from sneakers_ml.features import crop_image
from sneakers_ml.models.onnx import predict_sklearn_onnx


def get_sift(image: Image.Image, sift_instance: type) -> np.ndarray:
    image_resized = image.resize((256, 256))
    image_cropped = crop_image(image_resized, 224, 224)
    _, descriptors = sift_instance.detectAndCompute(np.array(image_cropped), None)
    return descriptors


def get_sift_descriptors(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[np.ndarray]]:
    dataset = ImageFolder(folder)
    sift_instance = cv2.SIFT_create()

    images_descriptors = []
    flattened_sift_descriptors = []
    for image, _ in tqdm(dataset):
        descriptors = get_sift(image, sift_instance)
        images_descriptors.append(descriptors)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx

    for image_descriptors in tqdm(images_descriptors):
        if image_descriptors is not None:
            for descriptor in image_descriptors:
                flattened_sift_descriptors.append(descriptor)

    return np.array(flattened_sift_descriptors), classes, class_to_idx, images_descriptors


def train_kmeans(n_clusters: int, flattened_sift_descriptors: np.ndarray) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    kmeans.fit(flattened_sift_descriptors)
    return kmeans


def get_feature_vector(sift_image_descriptors: np.ndarray, kmeans: KMeans) -> np.ndarray:
    n_clusters = kmeans.get_params()["n_clusters"]
    image_feature_vector = [0] * n_clusters
    if sift_image_descriptors is not None:
        descriptors_clusters = kmeans.predict(sift_image_descriptors)
        for cluster in descriptors_clusters:
            image_feature_vector[cluster] += 1
    return np.array(image_feature_vector)


def get_sift_features(
    folder: str, n_clusters: int, kmeans: Union[KMeans, None] = None
) -> tuple[np.ndarray, np.ndarray, dict[str, int], KMeans, np.ndarray]:
    flattened_sift_descriptors, classes, class_to_idx, images_descriptors = get_sift_descriptors(folder)
    kmeans = kmeans if kmeans is not None else train_kmeans(n_clusters, flattened_sift_descriptors)
    features = [get_feature_vector(images_descriptors, kmeans) for images_descriptors in tqdm(images_descriptors)]

    return np.array(features), classes, class_to_idx, kmeans, flattened_sift_descriptors[:1]


def get_sift_feature(image: Image.Image, kmeans: rt.InferenceSession, n_clusters: int) -> np.array:
    sift_instance = cv2.SIFT_create()
    descriptors = get_sift(image, sift_instance)

    image_feature_vector = [0] * n_clusters
    if descriptors is not None:
        descriptors_clusters = predict_sklearn_onnx(kmeans, descriptors)
        for cluster in descriptors_clusters:
            image_feature_vector[cluster] += 1
    return np.array(image_feature_vector)

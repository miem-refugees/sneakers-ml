import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from sneakers_ml.features import crop_image


def get_sift_features(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dataset = ImageFolder(folder)

    features = []
    last_calculated: np.array = None
    for image, _ in tqdm(dataset):
        feature = get_sift(image)
        if feature is not None:
            features.append(feature.ravel())
            last_calculated = feature.ravel()
        else:
            print("None image", image.info)
            features.append(last_calculated)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx
    numpy_features = features

    return numpy_features, classes, class_to_idx


def get_sift(image: Image.Image) -> np.ndarray:
    image_resized = image.resize((256, 256))
    image_cropped = crop_image(image_resized, 224, 128)
    image_grey = cv2.cvtColor(np.asarray(image_cropped), cv2.COLOR_BGR2RGB)
    _, descriptor = cv2.SIFT_create().detectAndCompute(image_grey, None)
    if descriptor is None:
        # try to compute on original size
        _, descriptor = cv2.SIFT_create().detectAndCompute(
            cv2.cvtColor(np.asarray(image_resized), cv2.COLOR_BGR2RGB), None
        )

    return descriptor


def bag_of_features(features, centres, k=500) -> np.ndarray:
    vec = np.zeros((1, k))
    for i in range(features.shape[0]):
        feat = features[i]
        diff = np.tile(feat, (k, 1)) - centres
        dist = pow(((pow(diff, 2)).sum(axis=1)), 0.5)
        idx_dist = dist.argsort()
        idx = idx_dist[0]
        vec[0][idx] += 1
    return vec


def kmeans(
    self,
    features: np.ndarray,
    k=10,
    criteria: tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
    flags: int = cv2.KMEANS_RANDOM_CENTERS,
) -> np.ndarray:
    _, _, centres = cv2.kmeans(features, k, None, criteria, 10, flags)
    img_vec = self.bag_of_features(features, centres, k)
    return img_vec

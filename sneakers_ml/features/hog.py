import numpy as np
from PIL import Image
from skimage.feature import hog
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from sneakers_ml.features import crop_image


def get_hog(image: Image.Image) -> np.ndarray:
    image_resized = image.resize((256, 256))
    image_cropped = crop_image(image_resized, 224, 128)

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


def get_hog_features(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dataset = ImageFolder(folder)

    features = []
    for image, _ in tqdm(dataset):
        feature = get_hog(image)
        features.append(feature)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx
    numpy_features = np.array(features)

    return numpy_features, classes, class_to_idx

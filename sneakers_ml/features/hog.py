import numpy as np
from skimage.feature import hog
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


def get_hog_features(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dataset = ImageFolder(folder)

    features = []
    for image, _ in tqdm(dataset):
        image_resized = image.resize((256, 256))

        width, height = image_resized.size
        crop_sides = 224
        crop_top_bot = 128
        left = (width - crop_sides) / 2
        top = (height - crop_top_bot) / 2
        right = (width + crop_sides) / 2
        bottom = (height + crop_top_bot) / 2

        image_cropped = image_resized.crop((left, top, right, bottom))

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
        features.append(feature)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx
    numpy_features = np.array(features)

    return numpy_features, classes, class_to_idx

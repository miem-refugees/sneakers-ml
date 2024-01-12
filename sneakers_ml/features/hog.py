import numpy as np
from PIL import Image
from skimage.feature import hog
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


def crop_image(image: Image.Image, crop_sides: int, crop_top_bot: int) -> Image.Image:
    width, height = image.size
    left = (width - crop_sides) // 2
    top = (height - crop_top_bot) // 2
    right = (width + crop_sides) // 2
    bottom = (height + crop_top_bot) // 2

    return image.crop((left, top, right, bottom))


def get_hog_features(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dataset = ImageFolder(folder)

    features = []
    for image, _ in tqdm(dataset):
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
        features.append(feature)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx
    numpy_features = np.array(features)

    return numpy_features, classes, class_to_idx

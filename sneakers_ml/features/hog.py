from pathlib import Path

import numpy as np
from skimage.feature import hog
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm.auto import tqdm


def get_hog_features(folder: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    transforms = v2.Compose(
        [
            v2.Resize((256, 256)),
            v2.CenterCrop((224, 224)),
        ]
    )

    dataset = ImageFolder(folder, transform=transforms)

    features = []
    for image, _ in tqdm(dataset):
        feature = hog(
            image,
            orientations=8,
            pixels_per_cell=(2, 2),
            cells_per_block=(1, 1),
            visualize=False,
            channel_axis=-1,
            feature_vector=True,
        )
        features.append(feature)

    classes = np.array(dataset.imgs)
    class_to_idx = dataset.class_to_idx
    features = np.array(features)

    return features, classes, class_to_idx

import pickle
from pathlib import Path

import numpy as np


def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as save_file:
        pickle.dump((numpy_features, classes, class_to_idx), save_file)

import logging
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sneakers_ml.features.hog import get_hog
from sneakers_ml.features.resnet152 import get_resnet152_feature
from sneakers_ml.features.sift import get_sift_feature
from sneakers_ml.models.onnx import load_catboost_onnx, load_sklearn_onnx, predict_catboost_onnx, predict_sklearn_onnx


class BrandClassifier:
    def __init__(self, config_path: Path):
        self._logger = logging.getLogger()

        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
        _models = config["models"]

        start_time = time.time()
        self._logger.info("Loading models: %s", ", ".join(_models))
        self.models = {}
        for model_category, models in _models.items():
            if model_category.startswith("sklearn"):
                for model in models:
                    self.models[Path(model).stem] = load_sklearn_onnx(model)
            elif model_category.startswith("catboost"):
                for model in models:
                    self.models[Path(model).stem] = load_catboost_onnx(model)

        self.kmeans = load_sklearn_onnx(config["features"]["kmeans"])
        self.labels = config["labels"]
        if not all(isinstance(num, int) and isinstance(label, str) for num, label in self.labels.items()):
            raise ValueError("Labels must be a dictionary of int: str:\n", self.labels)

        end_time = time.time()
        self._logger.info("All models loaded in %.2f seconds", end_time - start_time)

    def predict_using_all_models(self, image: Image.Image) -> dict:
        preds = {}
        hog_embedding = get_hog(image)[np.newaxis]
        resnet_embedding = get_resnet152_feature(image)[np.newaxis]
        sift_embedding = get_sift_feature(image, self.kmeans, 2000)[np.newaxis]
        for model in self.models:
            if "hog" in model:
                if "catboost" in model:
                    preds[model] = predict_catboost_onnx(self.models[model], hog_embedding)[0][0]
                else:
                    preds[model] = predict_sklearn_onnx(self.models[model], hog_embedding)[0]
            elif "resnet" in model:
                if "catboost" in model:
                    preds[model] = predict_catboost_onnx(self.models[model], resnet_embedding)[0][0]
                else:
                    preds[model] = predict_sklearn_onnx(self.models[model], resnet_embedding)[0]
            elif "sift" in model:
                if "catboost" in model:
                    preds[model] = predict_catboost_onnx(self.models[model], sift_embedding)[0][0]
                else:
                    preds[model] = predict_sklearn_onnx(self.models[model], sift_embedding)[0]
        return {model: self.labels.get(int(prediction)) for model, prediction in preds.items()}

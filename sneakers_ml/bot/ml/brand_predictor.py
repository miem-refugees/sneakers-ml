import time
from pathlib import Path

import numpy as np
from PIL import Image

from sneakers_ml.bot.loggers import ml as ml_logger
from sneakers_ml.features.hog import get_hog
from sneakers_ml.features.resnet152 import get_resnet152_feature
from sneakers_ml.models.onnx import load_catboost_onnx, load_sklearn_onnx, predict_catboost_onnx, predict_sklearn_onnx


class BrandPredictor:
    def __init__(self):
        self.models = {}
        self.logger = ml_logger

        sklearn_hog_models = [
            "data/models/brands-classification/hog-sgd.onnx",
            "data/models/brands-classification/hog-svc.onnx",
        ]
        sklearn_resnet_models = [
            "data/models/brands-classification/resnet-sgd.onnx",
            "data/models/brands-classification/resnet-svc.onnx",
        ]
        catboost_hog_models = ["data/models/brands-classification/hog-catboost.onnx"]
        catboost_resnet_models = ["data/models/brands-classification/resnet-catboost.onnx"]

        self.logger.debug("Loading sklearn_hog_models...")
        for model in sklearn_hog_models + sklearn_resnet_models:
            self.models[Path(model).stem] = load_sklearn_onnx(model)

        self.logger.debug("Loading catboost_hog_models...")
        for model in catboost_hog_models + catboost_resnet_models:
            self.models[Path(model).stem] = load_catboost_onnx(model)

        self.logger.info("All models loaded")
        self.labels = {
            0: "adidas",
            1: "asics",
            2: "clarks",
            3: "converse",
            4: "jordan",
            5: "kangaroos",
            6: "karhu",
            7: "new balance",
            8: "nike",
            9: "puma",
            10: "reebok",
            11: "saucony",
            12: "vans",
        }

    def predict(self, image: Image.Image) -> dict:
        """
        TODO: improve
        """
        start_time = time.time()
        preds = {}
        self.logger.debug("Calculating hog embedding...")
        hog_embedding = get_hog(image)[np.newaxis]
        self.logger.debug("Calculating resnet embedding...")
        resnet_embedding = get_resnet152_feature(image)[np.newaxis]
        for model in self.models:
            self.logger.debug(f"Predicting {model}...")
            if "hog" in model:
                if "catboost" in model:
                    preds[model] = self.label_decode(predict_catboost_onnx(self.models[model], hog_embedding)[0][0])
                else:
                    preds[model] = self.label_decode(predict_sklearn_onnx(self.models[model], hog_embedding)[0])
            else:
                if "catboost" in model:
                    preds[model] = self.label_decode(predict_catboost_onnx(self.models[model], resnet_embedding)[0][0])
                else:
                    preds[model] = self.label_decode(predict_sklearn_onnx(self.models[model], resnet_embedding)[0])
        self.logger.info("Predictions done in %d sec", (time.time() - start_time))
        # TODO: write metrics
        return preds

    def label_decode(self, label: str) -> str:
        return self.labels.get(int(label))

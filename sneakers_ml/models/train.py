import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING, Union

import hydra
import numpy as np
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import save_model

if TYPE_CHECKING:
    from catboost import CatBoostClassifier
    from sklearn.base import BaseEstimator

    from sneakers_ml.features.base import BaseFeatures


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train(cfg: DictConfig) -> None:
    np.random.seed(42)
    random.seed(42)

    results = [["model_name", *cfg.metrics.keys()]]

    for feature in cfg.features:
        tqdm.write(f"Using {feature}")
        feature_instance: BaseFeatures = instantiate(config=cfg.features[feature], config_data=cfg.data)
        x_train, x_val, x_test, y_train, y_val, y_test = feature_instance.load_train_val_test_splits()

        x_train_val = np.concatenate((x_train, x_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val))

        for model in tqdm(cfg.models):
            tqdm.write(f"Training {model}")
            model_instance: Union[BaseEstimator, CatBoostClassifier] = instantiate(cfg.models[model].model)
            model_instance.fit(x_train_val, y_train_val)
            pred = model_instance.predict(x_test)

            scores = [f"{feature}-{model}"]
            for metric in cfg.metrics:
                score = round(call(config=cfg.metrics[metric], y_true=y_test, y_pred=pred), 2)
                tqdm.write(f"{metric}: {score}")
                scores.append(score)

            results.append(scores)

            save_path = Path(cfg.paths.models_save) / f"{feature}-{model}.onnx"
            save_model(model_instance, x_train_val, str(save_path))

    results_save_path = Path(cfg.paths.results)
    if not results_save_path.exists():
        model = DummyClassifier()
        model.fit(x_train_val, y_train_val)
        pred = model.predict(x_test)

        scores = ["Baseline"]
        tqdm.write("DummyClassifier")
        for metric in cfg.metrics:
            score = round(call(config=cfg.metrics[metric], y_true=y_test, y_pred=pred), 2)
            tqdm.write(f"{metric}: {score}")
            scores.append(score)
        results.insert(1, scores)

    with results_save_path.open("a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(results)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter

# python sneakers_ml/models/train.py -m data=brands_classification,brands_classification_filtered

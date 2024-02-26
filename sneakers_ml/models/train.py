import csv
import random
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import save_model


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train(cfg: DictConfig) -> None:
    np.random.seed(42)
    random.seed(42)

    results = [["model_name", *cfg.metrics.keys()]]

    for feature in cfg.features:
        tqdm.write(f"Using {feature}")
        feature_instance = instantiate(config=cfg.features[feature], config_data=cfg.data)
        x_train, x_val, x_test, y_train, y_val, y_test = feature_instance.load_train_val_test_splits()

        x_train_val = np.concatenate((x_train, x_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val))

        num_samples = x_train_val.shape[0]

        # Choose 300 random indices
        random_indices = np.random.choice(num_samples, 100, replace=False)

        x_train_val = x_train_val[random_indices]
        y_train_val = y_train_val[random_indices]

        for model in tqdm(cfg.models):
            tqdm.write(f"Training {model}")
            model_instance = instantiate(cfg.models[model].model)
            model_instance.fit(x_train_val, y_train_val)
            pred = model_instance.predict(x_test)

            scores = [f"{feature}-{model}"]
            for metric in cfg.metrics:
                score = call(config=cfg.metrics[metric], y_true=y_test, y_pred=pred)
                tqdm.write(f"{metric}: {score}")
                scores.append(score)

            results.append(scores)

            save_path = Path(cfg.paths.models_save) / f"{feature}-{model}.onnx"
            save_model(model_instance, x_train_val, str(save_path))

    results_save_path = Path(cfg.paths.results)
    with results_save_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(results)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter

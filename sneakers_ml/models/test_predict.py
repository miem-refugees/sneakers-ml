import pytest
from hydra import compose, initialize
from loguru import logger
from PIL import Image

from sneakers_ml.models.predict import BrandsClassifier


class TestBrandClassifier:
    def setup_method(self) -> None:
        with initialize(version_base=None, config_path="../../config", job_name="predict"):
            self.config = compose(config_name="config")

    def test_brand_classifier(self) -> None:
        image = Image.new(mode="RGB", size=(500, 500))

        try:
            logger.info("Creating class instance...")
            self.classifier = BrandsClassifier(self.config)
        except Exception as e:
            pytest.fail(f"Exception raised: {e}")

        try:
            logger.info("Predicting single image...")
            prediction = self.classifier.predict([image])
        except Exception as e:
            pytest.fail(f"Exception raised: {e}")

        try:
            logger.info("Predicting multiple images...")
            prediction = self.classifier.predict([image, image, image])
        except Exception as e:
            pytest.fail(f"Exception raised: {e}")

        assert isinstance(prediction[0], dict)
        assert isinstance(prediction[1], dict)

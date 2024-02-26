# import logging

# import pytest
# from PIL import Image

# from sneakers_ml.models.old_code_brand_classifier.brand_classifier import BrandClassifier


# class TestBrandClassifier:
#     def setup_method(self):
#         logging.basicConfig(level=logging.INFO)
#         self.config = "configs/mlconfig.prod.yml"

#     def test_brand_classifier(self):
#         # smart way to get path of ../../../data/models.yml
#         classifier = BrandClassifier(self.config)
#         im = Image.new(mode="RGB", size=(200, 200))
#         try:
#             logging.info("Predicting...")
#             prediction = classifier.predict_using_all_models(im)
#         except Exception as e:
#             pytest.fail(f"Exception raised: {e}")
#         logging.info("Prediction: %s", prediction)
#         assert isinstance(prediction, dict)

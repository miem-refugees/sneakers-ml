import os
import pathlib
import time
from hashlib import md5
from typing import Annotated

from fastapi import APIRouter, Form, UploadFile
from hydra import compose, initialize
from loguru import logger
from PIL import Image

from sneakers_ml.app.config import config
from sneakers_ml.app.service.redis import RedisCache
from sneakers_ml.app.service.s3 import S3ImageUtility
from sneakers_ml.models.predict import BrandsClassifier

predictor: BrandsClassifier = None
s3 = S3ImageUtility("user_images")
redis = RedisCache(host=config.redis_host, port=config.redis_port)

router: APIRouter = APIRouter(prefix="/classify-brand", tags=["brand-classification"])


@router.on_event("startup")
async def load_model():
    config_relative_path = os.path.relpath((pathlib.Path.cwd() / config.ml_config_path), pathlib.Path(__file__).parent)
    logger.info("Loading models config. Resolved as: {}", config_relative_path)
    with initialize(version_base=None, config_path=config_relative_path, job_name="fastapi"):
        cfg = compose(config_name="config")
        global predictor
        predictor = BrandsClassifier(cfg)
    logger.info("Loaded BrandsClassifier")


@router.post("/upload/")
async def post_image_to_classify(
    image: UploadFile,
    # background_tasks: BackgroundTasks,
    username: Annotated[str, Form()],
):
    image.file.seek(0)
    image = Image.open(image.file)
    # background_tasks.add_task(s3.write_image_to_s3, image=image, name=f"{username}/{image.filename}")

    image_key = md5(image.tobytes()).hexdigest()  # nosec B324 (weak hash)
    cached_prediction = redis.get(image_key)
    if cached_prediction:
        logger.info("Found cached prediction for image: {}", image_key)
        return cached_prediction

    start_time = time.time()
    predictions = predictor.predict(images=[image])[1]
    end_time = time.time()
    logger.info("Predicted {} in {} seconds", predictions, end_time - start_time)

    redis.set(image_key, predictions, ttl=3600)
    return predictions

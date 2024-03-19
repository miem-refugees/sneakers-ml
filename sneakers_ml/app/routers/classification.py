import os
import pathlib
import time
from typing import Annotated

from fastapi import APIRouter, Form, UploadFile
from hydra import compose, initialize
from loguru import logger
from PIL import Image

from sneakers_ml.app.config import config
from sneakers_ml.app.service.s3 import S3ImageUtility
from sneakers_ml.models.predict import BrandsClassifier

predictor: BrandsClassifier = None
s3 = S3ImageUtility("user_images")

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
    f"{username}/{image.filename}"
    image = Image.open(image.file)
    # background_tasks.add_task(s3.write_image_to_s3, image=image, name=s3_key)

    start_time = time.time()
    prediction = predictor.predict(images=[image])[1]  # можно несколько картинок, но обязательно list
    end_time = time.time()
    logger.info("Predicted %s in %f seconds", prediction, end_time - start_time)
    return prediction

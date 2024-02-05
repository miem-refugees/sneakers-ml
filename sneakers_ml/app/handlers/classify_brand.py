import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, UploadFile  # , BackgroundTasks
from PIL import Image

from sneakers_ml.app.service.s3 import S3ImageUtility
from sneakers_ml.models.brand_classifier import BrandClassifier

predictor: BrandClassifier = None
s3 = S3ImageUtility("user_images")

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(prefix="/classify-brand", tags=["brand-classification"])


@router.on_event("startup")
async def load_model():
    global predictor
    predictor = BrandClassifier(Path(__file__).parent.parent.parent.parent / "configs/mlconfig.prod.yml")
    logger.info("Loaded BrandClassifier")


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
    prediction = predictor.predict_using_all_models(image=image)
    end_time = time.time()
    logger.info("Predicted %s in %f seconds", prediction, end_time - start_time)
    return prediction

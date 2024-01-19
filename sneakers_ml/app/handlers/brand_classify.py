import logging
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from PIL import Image

from sneakers_ml.app.models.brand_classify import BrandClassifyRequest
from sneakers_ml.app.service.s3 import S3ImageUtility
from sneakers_ml.models.brand_classifier import BrandClassifier

predictor: BrandClassifier = None
s3 = S3ImageUtility("user_images")

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(prefix="/classify-brand", tags=["image", "ml", "brand-classification"])


@router.on_event("startup")
async def load_model():
    global predictor
    predictor = BrandClassifier(Path(__file__).parent.parent.parent.parent / "configs/mlconfig.prod.yml")
    logger.info("Loaded BrandClassifier")


@router.post("/")
async def post_image_to_classify(request: BrandClassifyRequest, background_tasks: BackgroundTasks):
    image = Image.frombytes(request.image)
    s3_key = f"{request.from_username}/{request.name}"
    background_tasks.add_task(s3.write_image_to_s3, image=image, name=s3_key)

    start_time = time.time()
    prediction = predictor.predict_using_all_models(image=image)
    end_time = time.time()
    logger.info("Predicted %s in %f seconds", prediction, end_time - start_time)
    return prediction

from pathlib import Path

from fastapi import APIRouter

from sneakers_ml.models.brand_classifier import BrandClassifier

predictor = None

brand_classifier_router: APIRouter = APIRouter(
    prefix="/v1/classify-brand", tags=["image", "ml", "brand-classification"]
)


@brand_classifier_router.on_startup()
async def load_model():
    global predictor
    predictor = BrandClassifier(Path(__file__).parent.parent.parent / "data/models/brands-classification")

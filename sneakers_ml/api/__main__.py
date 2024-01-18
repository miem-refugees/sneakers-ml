from fastapi import FastAPI

from sneakers_ml.api.configs.environment import get_environment_variables
from sneakers_ml.api.routers.v1.brand_classifier import brand_classifier_router

# Application Environment Configuration
env = get_environment_variables()

# Core Application Instance
app = FastAPI(
    title=env.APP_NAME,
    version=env.API_VERSION,
)

# Add Routers
app.include_router(brand_classifier_router)

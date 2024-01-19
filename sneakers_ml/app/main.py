from fastapi import FastAPI

from sneakers_ml.app.handlers import brand_classify_router, common_router

app = FastAPI(title="Sneakers ML backend", debug=True)

# Add Routers
app.include_router(common_router)
app.include_router(brand_classify_router)

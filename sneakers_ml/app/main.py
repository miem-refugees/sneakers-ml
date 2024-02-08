from fastapi import FastAPI

from sneakers_ml.app.handlers import classify_brand_router, common_router

app = FastAPI(title="Sneakers ML backend", debug=True)

# Add Routers
app.include_router(common_router)
app.include_router(classify_brand_router)

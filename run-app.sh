#!/bin/bash

set -e

# Pull the DVC data
dvc pull data/models/brands-classification.dvc -f

# Start the FastAPI app with Uvicorn
uvicorn sneakers_ml.app.main:app --proxy-headers --host 0.0.0.0 --port 8000

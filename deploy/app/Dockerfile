FROM nvidia/cuda:12.3.1-base-ubuntu22.04 as base

LABEL authors="Slava and Daniil, miem-refugees"

    # Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automaticly, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'

# Install Python and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python-is-python3 \
    curl \
    # for opencv
    ffmpeg libsm6 libxext6 \
    build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

WORKDIR /sneakers_ml

# install dependencies
COPY pyproject.toml poetry.lock .

RUN poetry config installer.max-workers 10
RUN --mount=type=cache,target=/root/.cache \
    poetry install --with api --no-interaction --no-ansi --no-root

# Setup DVC
COPY .dvc .dvc/
COPY .git .git/
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID}
RUN dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY}

COPY data data/ 
COPY configs configs/
COPY sneakers_ml sneakers_ml/

RUN dvc pull data/models/brands-classification.dvc -f -d

WORKDIR /
CMD ["uvicorn", "sneakers_ml.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

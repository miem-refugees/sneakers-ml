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
    curl wget && \
    # install dvc
    wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list && \
    wget -qO - https://dvc.org/deb/iterative.asc | gpg --dearmor > packages.iterative.gpg && \
    install -o root -g root -m 644 packages.iterative.gpg /etc/apt/trusted.gpg.d/ && \
    rm -f packages.iterative.gpg && \
    apt update && apt-get install -y --no-install-recommends \
    dvc \
    # for opencv
    ffmpeg libsm6 libxext6 \
    build-essential && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/; \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /sneakers_ml

# install dependencies
COPY pyproject.toml poetry.lock ./

RUN poetry install --with api --no-interaction --no-ansi --no-root

COPY . ./

# Setup DVC
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID} &&  \
    dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    dvc pull data/models/brands-classification.dvc -f -d

EXPOSE 80
CMD ["uvicorn", "sneakers_ml.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]

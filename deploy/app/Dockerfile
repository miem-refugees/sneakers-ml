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
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    \
    # this is where our requirements + virtual environment will live
    VIRTUAL_ENV="/venv"

# Update the PATH
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"

ENV AWS_ACCESS_KEY_ID AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY AWS_SECRET_ACCESS_KEY

# Install Python and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python-is-python3 \
    curl \
    wget \
    # for opencv
    ffmpeg libsm6 libxext6 \
    build-essential

# Setup data from DVC (using aws S3)
ADD deploy/app/aws_setup.sh /root/aws_setup.sh
ADD deploy/app/dvc_setup.sh /root/dvc_setup.sh

RUN /root/aws_setup.sh
RUN /root/dvc_setup.sh

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

WORKDIR /sneakers_ml
ENV PYTHONPATH="/sneakers-ml:$PYTHONPATH"

# install dependencies
COPY pyproject.toml poetry.lock /sneakers_ml/

RUN poetry config installer.max-workers 10
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --with api --no-interaction --no-ansi --no-root

COPY data sneakers_ml configs /sneakers_ml/

RUN pip install dvc && \
    dvc pull data/models/brands-classification.dvc && \
    pip uninstall dvc

WORKDIR /
CMD ["uvicorn", "sneakers_ml.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

FROM python:3.9-slim as python-base

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
    \
    # this is where our requirements + virtual environment will live
    VIRTUAL_ENV="/venv"

ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV

WORKDIR /sneakers_ml
ENV PYTHONPATH="/sneakers-ml:$PYTHONPATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

COPY pyproject.toml poetry.lock sneakers_ml configs /sneakers_ml/

FROM python-base as api-service

RUN sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples

RUN POETRY_INSTALLER_MAX_WORKERS=16  \
    poetry install --with api --without ml --no-interaction --no-ansi

WORKDIR /
CMD ["python", "sneakers_ml/api/main.py"]

FROM python-base as telegram-bot-service

RUN POETRY_INSTALLER_MAX_WORKERS=16  \
    poetry install --only bot --no-interaction --no-ansi

WORKDIR /
CMD ["python", "sneakers_ml/bot/main.py"]

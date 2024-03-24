FROM ksusonic/ubuntu-cuda-poetry:v0.0.1

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

WORKDIR /sneakers_ml

# install dependencies
COPY pyproject.toml poetry.lock ./

RUN poetry install --with api --no-interaction --no-ansi --no-root

COPY . ./

# Setup DVC
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID} &&  \
    dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY}

EXPOSE 8000
ENTRYPOINT [ "./run-app.sh" ]

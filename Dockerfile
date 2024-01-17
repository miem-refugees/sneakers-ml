FROM python:3.9-slim as production

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PATH "/app/scripts:${PATH}"

WORKDIR /app

# Install Poetry
RUN set +x \
 && apt update \
 && apt upgrade -y \
 && apt install -y curl gcc build-essential \
 && curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python -\
 && cd /usr/local/bin \
 && ln -s /opt/poetry/bin/poetry \
 && poetry config virtualenvs.create false \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock sneakers_ml/bot /app/
RUN poetry config installer.max-workers 16
RUN --mount=type=cache,target=/root/.cache poetry install --with telegram --without ml --no-interaction --no-ansi -vvv

# ML models
RUN dvc pull data/models/brands-classification.dvc
ADD data/models/brands-classification /app/data/models/brands-classification

ADD . /app/
CMD ["python", "__main__.py"]

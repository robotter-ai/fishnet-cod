FROM python:3.11-bullseye
LABEL authors="mhh"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
COPY ./src ./src
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["poetry", "run", "uvicorn", "src.fishnet_cod.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
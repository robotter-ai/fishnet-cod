# Stage 1: Base Image
FROM python:3.11 AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock README.md ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

RUN ls -a

COPY fishnet_cod ./fishnet_cod

CMD ["echo", "Base image for fishnet services"]

FROM base AS apiservice

CMD ["poetry", "run", "uvicorn", "fishnet_cod.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS listenerservice

CMD ["poetry", "run", "python3", "-m", "fishnet_cod.local_listener"]
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app/

RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    build-essential \
    python3-dev \
    git

COPY . /app/

ENV UV_COMPILE_BYTECODE=1

RUN uv venv && \
    export PATH="/app/.venv/bin:$PATH" && \
    uv sync --frozen

FROM python:3.13-slim

WORKDIR /app/

RUN groupadd user && useradd --home-dir /app -g user user && chown -R user:user /app

COPY --from=builder --chown=user:user /app/.venv /app/.venv
COPY --chown=user:user llm_bot /app/llm_bot
COPY --chown=user:user README.md app.py LICENSE.md openapi3_1.yml /app/

USER user

ENV PYTHONOPTIMIZE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV GRANIAN_THREADS=2
ENV GRANIAN_WORKERS=1
ENV GRANIAN_WORKERS_MAX_RSS=512
ENV GRANIAN_BLOCKING_THREADS=1
ENV GRANIAN_INTERFACE=asgi
ENV GRANIAN_HOST=0.0.0.0
ENV GRANIAN_LOG_ACCESS_ENABLED=1

EXPOSE 8000

CMD ["granian", "app"]

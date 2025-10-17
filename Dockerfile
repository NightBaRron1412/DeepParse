FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip install --upgrade pip && \
    pip install uv && \
    uv sync

ENV PYTHONPATH=/workspace
ENTRYPOINT ["python", "-m", "deepparse.cli"]

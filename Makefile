.PHONY: setup lint test demo tables docker

PYTHON?=python3
VENV?=.venv

setup:
uv sync

lint:
uv run ruff check deepparse tests

test:
uv run pytest -q

_demo_common:
./scripts/prepare_paths.sh
uv run python -m deepparse.cli synth --config configs/demo_small.yaml --mode offline
uv run python -m deepparse.cli eval --config configs/demo_small.yaml --deterministic

demo: _demo_common
uv run python -m deepparse.cli table --inputs artifacts/outputs/demo_metrics.csv --out artifacts/outputs/tables/


tables:
./scripts/regenerate_tables.sh

docker:
docker build -t deepparse-artifact .

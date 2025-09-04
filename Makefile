

.PHONY: train dev

dev:
	uv sync
	uv venv && source .venv/bin/activate

train:
	uv run python -m src.train 
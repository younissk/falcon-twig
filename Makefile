

.PHONY: train dev

dev:
	uv sync
	uv venv && source .venv/bin/activate

train:
	uv sync
	uv run train src/train.py --config src/config.toml
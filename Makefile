

.PHONY: train dev

dev:
	uv sync
	uv venv && source .venv/bin/activate

train:
	uv run src/train.py 


.PHONY: train dev

dev:
	uv sync
	uv venv && source .venv/bin/activate

train:
	tmux new-session -s "falcon-twig" -n "train"
	uv run python -m src.train 
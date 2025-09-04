

.PHONY: train dev

dev:
	uv sync
	uv venv && source .venv/bin/activate
	tmux new-session -s "falcon-twig" -n "train"

train:
	uv run python -m src.train 

upload:
	hf auth whoami
	hf upload younissk/Falcon-Twig-7B outputs

make all:
	make train
	make upload
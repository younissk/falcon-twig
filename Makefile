

.PHONY: train dev test-env fix-env doctor doctor-fix train311 install-mamba

dev:
	uv sync
	uv venv && source .venv/bin/activate

tmux:
	tmux new-session -s "falcon-twig" -n "train"

tmux-attach:
	tmux attach-session -t "falcon-twig"

tmux-kill:
	tmux kill-session -t "falcon-twig"

train:
	uv run python -m src.train 

upload:
	hf auth whoami
	hf upload younissk/Falcon-Twig-7B outputs

make all:
	make train
	make upload

# Diagnose CUDA/Torch/Mamba fastpath environment
doctor:
	bash scripts/fastpath_doctor.sh diagnose

# Attempt automated fix: set up Python 3.11 + install fastpath extras
doctor-fix:
	TARGET_PYTHON=3.11 INSTALL_FLASH=0 bash scripts/fastpath_doctor.sh fix

# Train with Python 3.11 explicitly
train311:
	UV_PYTHON=3.11 uv run python -m src.train

# Install Mamba SSM
install-mamba:
	pip install causal-conv1d>=1.4.0
	pip install mamba-ssm
	pip install mamba-ssm[dev]
	uv pip install causal-conv1d>=1.4.0 
	uv pip install mamba-ssm
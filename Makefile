

.PHONY: train dev test-env fix-env

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

test-env:
	@echo "=== Falcon-Twig Environment Analysis ==="
	@echo ""
	@echo "1. Python Version:"
	@uv run python --version
	@echo ""
	@echo "2. PyTorch & CUDA Status:"
	@uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
	@echo ""
	@echo "3. GPU Info (if available):"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@echo "4. bitsandbytes Status:"
	@uv run python -c "try: import bitsandbytes as bnb; print('✓ bitsandbytes available:', bnb.__version__); except ImportError as e: print('✗ bitsandbytes not available:', str(e))"
	@echo ""
	@echo "5. Flash Attention Status:"
	@uv run python -c "try: import flash_attn; print('✓ flash-attn available:', flash_attn.__version__); except ImportError as e: print('✗ flash-attn not available:', str(e))"
	@echo ""
	@echo "6. Mamba SSM Fast Kernels Status:"
	@uv run python -c "try: from mamba_ssm.ops.selective_state_update import selective_state_update; from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; print('✓ Mamba fast kernels available'); print(f'  selective_state_update: {selective_state_update is not None}'); print(f'  causal_conv1d_fn: {causal_conv1d_fn is not None}'); print(f'  causal_conv1d_update: {causal_conv1d_update is not None}'); except ImportError as e: print('✗ Mamba fast kernels not available:', str(e))"
	@echo ""
	@echo "7. Current Environment Analysis:"
	@uv run python -c "import sys; print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); print(f'Platform: {sys.platform}'); import torch; print(f'PyTorch CUDA build: {torch.cuda.is_available()}')"
	@echo ""
	@echo "=== Recommendations ==="
	@uv run python -c "import sys; py_ver = sys.version_info; if py_ver >= (3, 12): print('⚠️  Python 3.12+ detected. Fast kernels require Python 3.11 or manual build from source.'); else: print('✓ Python version compatible with prebuilt wheels')"
	@uv run python -c "import torch; print('✓ CUDA PyTorch detected' if torch.cuda.is_available() else '⚠️  CPU-only PyTorch detected. Install CUDA version for GPU training.')"

fix-env:
	@echo "=== Fixing Environment for Optimal Performance ==="
	@echo ""
	@echo "Step 1: Installing Python 3.11 and setting up environment..."
	@UV_PYTHON=3.11 uv python install 3.11
	@echo ""
	@echo "Step 2: Installing CUDA PyTorch (if needed)..."
	@UV_PYTHON=3.11 uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || (echo "Installing CUDA PyTorch..." && UV_PYTHON=3.11 uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio)
	@echo ""
	@echo "Step 3: Installing fast kernels and optimizations..."
	@UV_PYTHON=3.11 uv sync -E cuda -E fastpath
	@echo ""
	@echo "Step 4: Verifying installation..."
	@UV_PYTHON=3.11 uv run python -c "from mamba_ssm.ops.selective_state_update import selective_state_update; from causal_conv1d import causal_conv1d_fn; import torch; print('✓ All fast kernels loaded successfully'); print(f'CUDA: {torch.cuda.is_available()}, Mamba: {selective_state_update is not None}, CausalConv1d: {causal_conv1d_fn is not None}')"
	@echo ""
	@echo "Step 5: Testing training setup..."
	@UV_PYTHON=3.11 uv run python -c "from src.modeling import load_model_4bit; from src.config import config; print('✓ Model loading functions available')"
	@echo ""
	@echo "=== Environment Fixed! ==="
	@echo "Run 'UV_PYTHON=3.11 make train' to start training with fast kernels."

make all:
	make train
	make upload
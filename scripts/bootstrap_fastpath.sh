#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Linux + NVIDIA GPU instance for Falcon-Twig training
# - Creates a Python 3.11 uv venv
# - Installs matching CUDA Torch, Mamba SSM + causal-conv1d fast kernels
# - Optionally installs flash-attn
# - Verifies the fast path is available
#
# Usage:
#   bash scripts/bootstrap_fastpath.sh            # auto-detect CUDA tag and install
#
# Optional env vars:
#   CUDA_TAG=cu126   # one of: cu128|cu126|cu124|cu121 (auto-detected if unset)
#   INSTALL_FLASH=1  # also install flash-attn (recommended)

OS_NAME="$(uname -s)"
if [[ "${OS_NAME}" != "Linux" ]]; then
  echo "[bootstrap] ERROR: Requires Linux + NVIDIA GPU" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] ERROR: nvidia-smi not found. GPU/driver not visible." >&2
  exit 1
fi

detect_cuda_tag() {
  local override_tag
  override_tag="${CUDA_TAG:-}"
  if [[ -n "$override_tag" ]]; then
    echo "$override_tag"; return 0
  fi
  local line major minor
  line=$(nvidia-smi | grep -o 'CUDA Version: [0-9][0-9]*\.[0-9][0-9]*' | head -n1 || true)
  if [[ -n "$line" ]]; then
    major=$(echo "$line" | awk '{print $3}' | cut -d. -f1)
    minor=$(echo "$line" | awk '{print $3}' | cut -d. -f2)
    if (( major == 12 )); then
      if   (( minor >= 8 )); then echo "cu128"; return 0
      elif (( minor >= 6 )); then echo "cu126"; return 0
      elif (( minor >= 4 )); then echo "cu124"; return 0
      elif (( minor >= 1 )); then echo "cu121"; return 0
      fi
    elif (( major == 11 )); then
      if (( minor >= 8 )); then echo "cu118"; return 0; fi
    fi
  fi
  echo "cu121"
}

TAG=$(detect_cuda_tag)
echo "[bootstrap] Using CUDA tag: ${TAG}"

export UV_PYTHON=3.11

echo "[bootstrap] Ensuring Python ${UV_PYTHON} installed for uv"
uv python install "${UV_PYTHON}" || true

echo "[bootstrap] Creating uv venv"
uv venv || true

echo "[bootstrap] Upgrading build tooling"
uv pip install -U pip setuptools wheel packaging ninja cmake

echo "[bootstrap] Installing CUDA Torch (${TAG})"
uv pip install --reinstall --index-url "https://download.pytorch.org/whl/${TAG}" torch torchvision torchaudio

echo "[bootstrap] Installing fast kernels (mamba-ssm, causal-conv1d)"
# A100 arch is sm80
export TORCH_CUDA_ARCH_LIST="8.0"
uv pip install --no-build-isolation --upgrade "mamba-ssm>=2.2.2,<3.0" "causal-conv1d>=1.4.0.post2"

if [[ "${INSTALL_FLASH:-1}" == "1" ]]; then
  echo "[bootstrap] Installing flash-attn (optional)"
  uv pip install --no-build-isolation --upgrade "flash-attn>=2.6.3"
else
  echo "[bootstrap] Skipping flash-attn install (INSTALL_FLASH=0)"
fi

echo "[bootstrap] Verifying fast path"
uv run python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
ok = True
try:
    import mamba_ssm
    print("mamba_ssm file:", mamba_ssm.__file__)
except Exception as e:
    ok = False
    print("mamba_ssm import error:", e)
try:
    from mamba_ssm.ops.selective_state_update import selective_state_update
    print("selective_state_update:", selective_state_update is not None)
    ok = ok and (selective_state_update is not None)
except Exception as e:
    ok = False
    print("selective_state_update import error:", e)
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("causal_conv1d_fn:", causal_conv1d_fn is not None)
    print("causal_conv1d_update:", causal_conv1d_update is not None)
    ok = ok and (causal_conv1d_fn is not None) and (causal_conv1d_update is not None)
except Exception as e:
    ok = False
    print("causal_conv1d import error:", e)
raise SystemExit(0 if ok else 2)
PY

echo "[bootstrap] Fast path verified"
echo "[bootstrap] Next: USE_4BIT=0 UV_PYTHON=3.11 uv run python -m src.train --config configs/first_run.toml"


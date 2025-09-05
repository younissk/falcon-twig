#!/usr/bin/env bash
set -euo pipefail

# Fastpath Doctor: diagnose and (optionally) fix fast CUDA kernel setup
# - Checks Torch CUDA, Mamba SSM, causal-conv1d, flash-attn, bitsandbytes
# - On fix: sets up a Python 3.11 env with uv and installs fastpath extras
#
# Usage:
#   bash scripts/fastpath_doctor.sh diagnose        # Only check and report
#   bash scripts/fastpath_doctor.sh fix             # Attempt to auto-fix (Linux/CUDA)
#
# Optional env vars:
#   TARGET_PYTHON=3.11     # Python version to target for fix (default: 3.11)
#   INSTALL_FLASH=1         # Also attempt flash-attn install on fix (Linux only)
#   VERBOSE=1               # Show verbose commands

if [[ "${VERBOSE:-}" == "1" ]]; then
  set -x
fi

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${HERE_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Ensure uv uses a workspace-local cache to avoid sandbox issues
export UV_CACHE_DIR="${ROOT_DIR}/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

ACTION="${1:-diagnose}"
TARGET_PYTHON="${TARGET_PYTHON:-3.11}"
INSTALL_FLASH="${INSTALL_FLASH:-0}"

log() { printf "[fastpath-doctor] %s\n" "$*"; }
warn() { printf "[fastpath-doctor][WARN] %s\n" "$*"; }
err() { printf "[fastpath-doctor][ERROR] %s\n" "$*" 1>&2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }
}

print_header() {
  echo "==== $* ===="
}

run_uv_python() {
  # Runs a small Python snippet inside the current uv project environment.
  # Accepts heredoc input on stdin.
  uv run python - "$@"
}

diagnose() {
  need_cmd uv

  print_header "System Info"
  uname -a || true
  if command -v lsb_release >/dev/null 2>&1; then lsb_release -a || true; fi
  echo "shell: $SHELL"
  echo "pwd: $(pwd)"
  echo

  print_header "GPU/Driver"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    warn "nvidia-smi not found (no NVIDIA GPU or drivers not installed)"
  fi
  echo

  print_header "Python/uv"
  uv --version || true
  # Try to show active Python inside uv env
  uv run python -c 'import sys; print("python:", sys.version.split()[0])' || true
  echo

  print_header "Torch/CUDA and Kernels"
  run_uv_python <<'PY' || true
import importlib, sys
try:
    import torch
    print('python:', sys.version.split()[0])
    print('torch:', torch.__version__)
    print('torch.version.cuda:', getattr(torch.version, 'cuda', None))
    print('cuda.is_available:', torch.cuda.is_available())
except Exception as e:
    print('torch import failed:', e)
    torch = None

def try_import(name):
    try:
        import importlib
        importlib.import_module(name)
        return True
    except Exception:
        return False

ok_mamba = try_import('mamba_ssm.ops.selective_state_update')
ok_causal = try_import('causal_conv1d')
ok_flash  = try_import('flash_attn')
ok_bnb    = try_import('bitsandbytes')
print('mamba_ssm.import:', ok_mamba)
print('causal_conv1d.import:', ok_causal)
print('flash_attn.import:', ok_flash)
print('bitsandbytes.import:', ok_bnb)

sel_ok = False
cc_fn_ok = False
cc_up_ok = False
if ok_mamba:
    try:
        from mamba_ssm.ops.selective_state_update import selective_state_update
        sel_ok = selective_state_update is not None
    except Exception as e:
        sel_ok = False
        print('mamba selective_state_update error:', e)
if ok_causal:
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        cc_fn_ok = causal_conv1d_fn is not None
        cc_up_ok = causal_conv1d_update is not None
    except Exception as e:
        cc_fn_ok = False
        cc_up_ok = False
        print('causal_conv1d ops error:', e)
print('mamba_ssm.selective_state_update:', sel_ok)
print('causal_conv1d.fn:', cc_fn_ok)
print('causal_conv1d.update:', cc_up_ok)
PY

  echo
  print_header "Summary"
  echo "- If torch.cuda.is_available is False or torch.version.cuda is None, you have a CPU-only PyTorch or no CUDA runtime."
  echo "- If mamba/causal_conv1d import True but ops False, the CUDA ops weren't built/found."
  echo "- On macOS, CUDA fastpath is not available. Use Linux + NVIDIA GPU."
}

fix() {
  need_cmd uv

  OS_NAME="$(uname -s)"
  if [[ "${OS_NAME}" != "Linux" ]]; then
    warn "Non-Linux system detected (${OS_NAME}). CUDA fastpath requires Linux + NVIDIA GPU."
  fi

  print_header "Ensuring Python ${TARGET_PYTHON} is available to uv"
  UV_PYTHON="${TARGET_PYTHON}" uv python install "${TARGET_PYTHON}" || true

  print_header "Syncing dependencies (extras: cuda, fastpath) with Python ${TARGET_PYTHON}"
  UV_PYTHON="${TARGET_PYTHON}" uv sync -E cuda -E fastpath || true

  print_header "Installing Mamba/causal-conv1d with no-build-isolation"
  # Force reinstall to bypass build isolation and catch missing wheels
  UV_PYTHON="${TARGET_PYTHON}" uv pip install --no-build-isolation --upgrade \
    "mamba-ssm>=2.2.2,<3.0" "causal-conv1d>=1.4.0.post2" || true

  if [[ "${INSTALL_FLASH}" == "1" && "${OS_NAME}" == "Linux" ]]; then
    print_header "Installing flash-attn (optional)"
    UV_PYTHON="${TARGET_PYTHON}" uv pip install --no-build-isolation --upgrade \
      "flash-attn>=2.6.3" || true
  else
    warn "Skipping flash-attn install (set INSTALL_FLASH=1 and ensure Linux/CUDA)"
  fi

  print_header "Verifying after install"
  UV_PYTHON="${TARGET_PYTHON}" uv run bash -lc 'python - <<"PY"\nimport torch\nprint("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", None), "avail:", torch.cuda.is_available())\ntry:\n  from mamba_ssm.ops.selective_state_update import selective_state_update\n  print("mamba selective_state_update:", selective_state_update is not None)\nexcept Exception as e:\n  print("mamba selective_state_update import failed:", e)\ntry:\n  from causal_conv1d import causal_conv1d_fn, causal_conv1d_update\n  print("causal_conv1d_fn:", causal_conv1d_fn is not None)\n  print("causal_conv1d_update:", causal_conv1d_update is not None)\nexcept Exception as e:\n  print("causal_conv1d import failed:", e)\ntry:\n  import flash_attn\n  print("flash_attn:", True)\nexcept Exception:\n  print("flash_attn:", False)\ntry:\n  import bitsandbytes as bnb\n  print("bitsandbytes:", True)\nexcept Exception:\n  print("bitsandbytes:", False)\nPY'

  echo
  print_header "Next Steps"
  echo "- If torch.cuda.is_available=False or cuda=None, install a CUDA-enabled PyTorch matching your system."
  echo "  Example (CUDA 12.1): UV_PYTHON=${TARGET_PYTHON} uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
  echo "- Then re-run: UV_PYTHON=${TARGET_PYTHON} make train"
}

case "${ACTION}" in
  diagnose)
    diagnose ;;
  fix)
    fix ;;
  *)
    err "Unknown action: ${ACTION}. Use: diagnose | fix"
    exit 2 ;;
esac

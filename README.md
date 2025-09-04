![Falcon-Twig](./images/Falcon-Twig-Banner.png)

# Falcon-Twig

Falcon-Twig is a fine-tuning project for a Falcon H1 model on Tool calling.

## Fast Training on A100

Falcon H1 uses Mamba SSM blocks that require CUDA kernels to be fast. If these
aren't installed, you'll see a warning like:

> The fast path is not available because one of (selective_state_update, causal_conv1d_fn, causal_conv1d_update) is None. Falling back to the naive implementation.

This fallback is 10–50x slower and leads to extremely long step times. To enable
the fast path on a CUDA Linux box (e.g., A100):

1) Install the GPU kernels (PyTorch must be a CUDA build):

```
uv pip install --no-build-isolation mamba-ssm>=2.2.2 causal-conv1d>=1.4.0.post2
# Optional but recommended for attention speedups where applicable
uv pip install --no-build-isolation flash-attn>=2.6.3
```

2) Verify kernels are visible:

```
uv run python - <<'PY'
from mamba_ssm.ops.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn
print('mamba selective_state_update:', selective_state_update is not None)
print('causal_conv1d_fn:', causal_conv1d_fn is not None)
PY
```

3) Train:

```
make train
```

Additional tips:
- Ensure CUDA is available (`python -c "import torch;print(torch.cuda.is_available())"`).
- On Ampere+, TF32 is enabled automatically by the trainer for faster matmuls.
- You can reduce `max_input_len`/`max_label_len` in `src/config.py` for faster iterations.

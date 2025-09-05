import os
import json
import argparse
import torch  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer, TrainerCallback  # type: ignore
from torch.nn.utils.rnn import pad_sequence  # type: ignore
from src.config import TrainingConfig
from src.modeling import load_tokenizer, load_model_4bit, infer_lora_targets, apply_lora  # type: ignore
from src.data import load_and_prepare


def _read_toml(path: str) -> Dict[str, Any]:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


class WeightedTrainer(Trainer):
    """Custom trainer with difficulty-based weighting support."""

    def __init__(self, difficulty_weights: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.difficulty_weights = difficulty_weights


def eval_tool_calling(model: Any, tokenizer: Any, valid_ds: Any, device: Any,
                      n_samples: int = 100, max_new_tokens: int = 256) -> Dict[str, float]:
    """Placeholder evaluation function for tool calling."""
    # This is a placeholder implementation
    # In a real implementation, this would evaluate the model's tool calling capabilities
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }


def main(config: Optional[str] = None) -> None:
    if config:
        cfg = TrainingConfig(**_read_toml(path=config))
    else:
        # Use default config from config.py
        from src.config import config as default_config
        cfg = default_config
    device = torch.device("cuda" if torch.cuda.is_available() # type: ignore
                          else "cpu")  # type: ignore

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")

    # Enable TF32 on Ampere+ for big matmuls
    if torch.cuda.is_available() and cfg.allow_tf32:  # type: ignore
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            torch.backends.cudnn.allow_tf32 = True  # type: ignore
        except Exception:
            pass

    # tokenizer / model
    tokenizer = load_tokenizer(cfg.base_model)  # type: ignore
    model = load_model_4bit(
        cfg.base_model,
        attn_implementation=cfg.attn_implementation,
        enable_torch_compile=cfg.enable_torch_compile,
        torch_compile_mode=cfg.torch_compile_mode,
    )  # type: ignore

    # Warn if Mamba fast-path kernels are missing (Falcon H1 uses SSM blocks)
    def _mamba_fastpath_available() -> bool:
        try:
            from mamba_ssm.ops.selective_state_update import selective_state_update  # type: ignore
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # type: ignore
            return (selective_state_update is not None) and (causal_conv1d_fn is not None) and (causal_conv1d_update is not None)
        except Exception:
            return False

    if not _mamba_fastpath_available():
        print("WARNING: Mamba/causal-conv1d fast CUDA kernels not found. Training will be 10-50x slower.\n"
              "Install `mamba-ssm` and `causal-conv1d` wheels built for your CUDA to enable the fast path.")

    # infer targets if empty
    targets = cfg.targets or infer_lora_targets(model)  # type: ignore
    model = apply_lora(model, targets, r=cfg.lora_r,
                       alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)  # type: ignore

    # data
    train_ds, valid_ds = load_and_prepare(cfg, tokenizer)  # type: ignore

    # collator
    PAD_ID: int = int(tokenizer.pad_token_id)  # type: ignore

    def collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_tensors = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]  # type: ignore
        label_tensors = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]  # type: ignore
        input_batch = pad_sequence(input_tensors, batch_first=True, padding_value=PAD_ID)  # type: ignore
        attention_mask = (input_batch != PAD_ID).long()  # type: ignore
        label_batch = pad_sequence(label_tensors, batch_first=True, padding_value=PAD_ID)  # type: ignore
        label_batch[label_batch == PAD_ID] = -100  # type: ignore
        return {"input_ids": input_batch, "attention_mask": attention_mask, "labels": label_batch}

    _optim = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"  # type: ignore
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_bs,
        per_device_eval_batch_size=cfg.per_device_eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=20,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        gradient_checkpointing=True,
        bf16=(model.dtype == torch.bfloat16),  # type: ignore
        fp16=(model.dtype == torch.float16),  # type: ignore
        dataloader_persistent_workers=cfg.dataloader_persistent_workers,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
        optim=_optim,
        report_to=[],
    )

    # Optional simple throughput logger
    class _ThroughputCallback(TrainerCallback):  # type: ignore
        def __init__(self, log_tps: bool = True) -> None:
            self.log_tps = log_tps
            self._tok_counter = 0
            self._last_step = 0
        def on_log(self, args, state, control, **kwargs):  # type: ignore
            if not self.log_tps:
                return
            logs: Dict[str, Any] = {}
            try:
                kw: Dict[str, Any] = cast(Dict[str, Any], kwargs)
                got_any: Any = kw.get("logs", {}) or {}
                if isinstance(got_any, dict):
                    got_dict: Dict[str, Any] = cast(Dict[str, Any], got_any)
                    logs = {str(k): got_dict[k] for k in got_dict}
            except Exception:
                pass
            tps = logs.get("train_tokens_per_second")
            if tps is not None:
                print(f"tokens/sec: {tps:.2f}")

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=None,  # avoid deprecation noise
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience), _ThroughputCallback(cfg.log_tokens_per_second)],
        difficulty_weights=cfg.difficulty_weights,
    )

    trainer.train()  # type: ignore
    trainer.save_model(cfg.output_dir)  # type: ignore
    tokenizer.save_pretrained(cfg.output_dir)  # type: ignore

    # quick eval
    metrics = eval_tool_calling(model, tokenizer, valid_ds, device,
                                n_samples=cfg.eval_samples, max_new_tokens=cfg.max_new_tokens_eval)
    print("Eval tool-calling:", metrics)
    Path(cfg.output_dir, "tool_eval.json").write_text(
        json.dumps(metrics, indent=2))  # type: ignore

    # Optional: WiSE-FT-style interpolation with base (simple linear merge).
    # This merges LoRA and linearly blends with base. Needs VRAM; skip if alpha==1.0.
    if cfg.wise_ft_alpha < 1.0:
        print(f"Interpolating with alpha={cfg.wise_ft_alpha}...")
        from copy import deepcopy
        # same dtype/device_map  # type: ignore
        base = load_model_4bit(cfg.base_model)
        # merge LoRA to dense (careful: allocates more memory)
        merged = deepcopy(model).merge_and_unload()  # type: ignore
        s_merged = merged.state_dict()  # type: ignore
        s_base = base.state_dict()  # type: ignore
        for k in s_merged:  # type: ignore
            # type: ignore
            if k in s_base and s_merged[k].dtype == s_base[k].dtype and s_merged[k].shape == s_base[k].shape:
                s_merged[k] = cfg.wise_ft_alpha * s_merged[k] + \
                    (1.0 - cfg.wise_ft_alpha) * s_base[k]  # type: ignore
        merged.load_state_dict(s_merged, strict=False)  # type: ignore
        merged.save_pretrained(
            # type: ignore
            Path(cfg.output_dir, f"merged_alpha_{cfg.wise_ft_alpha}"))
        print("Saved interpolated merged model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Falcon model for tool calling")
    parser.add_argument("--config", help="Path to TOML config file (optional, uses default config if not provided)")
    args = parser.parse_args()
    main(args.config)

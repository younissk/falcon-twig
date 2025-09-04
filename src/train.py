import os
import json
import argparse
import torch  # type: ignore
from pathlib import Path
from typing import Any, Dict, List
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer  # type: ignore
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


def main(config: str) -> None:
    cfg = TrainingConfig(**_read_toml(path=config))
    device = torch.device("cuda" if torch.cuda.is_available() # type: ignore
                          else "cpu")  # type: ignore

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")

    # tokenizer / model
    tokenizer = load_tokenizer(cfg.base_model)  # type: ignore
    model = load_model_4bit(cfg.base_model)  # type: ignore

    # infer targets if empty
    targets = cfg.targets or infer_lora_targets(model)  # type: ignore
    model = apply_lora(model, targets, r=cfg.lora_r,
                       alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)  # type: ignore

    # data
    train_ds, valid_ds = load_and_prepare(cfg, tokenizer)  # type: ignore

    # collator
    PAD_ID = tokenizer.pad_token_id  # type: ignore

    def collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch  # type: ignore
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long)
                     for f in features]  # type: ignore
        labels = [torch.tensor(f["labels"],    dtype=torch.long)
                  for f in features]  # type: ignore
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD_ID)  # type: ignore
        attention_mask = (input_ids != PAD_ID).long()  # type: ignore
        labels = torch.nn.utils.rnn.pad_sequence( # type: ignore
            labels, batch_first=True, padding_value=PAD_ID)  # type: ignore
        labels[labels == PAD_ID] = -100  # type: ignore
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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
        eval_strategy="steps",  # Fixed parameter name
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
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=None,  # avoid deprecation noise
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        difficulty_weights=cfg.difficulty_weights,
    )

    _ = trainer.train()  # type: ignore
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
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    args = parser.parse_args()
    main(args.config)

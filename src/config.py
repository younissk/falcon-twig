"""Configuration for the Falcon-Twig fine-tuning project."""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    dataset_id: str = Field(
        default="younissk/tool-calling-mix",
        description="Hugging Face dataset ID to use for training"
    )

    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    num_train_epochs: int = Field(
        default=3,
        description="Number of training epochs"
    )

    lora_r: int = Field(
        default=16,
        description="Rank of the LoRA matrix"
    )
    lora_alpha: int = Field(
        default=32,
        description="Alpha of the LoRA matrix"
    )
    lora_dropout: float = Field(
        default=0.05,
        description="Dropout of the LoRA matrix"
    )
    targets: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
        description="Targets of the LoRA matrix"
    )

    epochs: int = Field(
        default=1,
        description="Number of training epochs"
    )
    
    learning_rate: float = Field(
        default=1e-4,
        description="Learning rate for the optimizer"
    )
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for the optimizer"
    )
    warmup_ratio: float = Field(
        default=0.1,
        description="Warmup ratio for the optimizer"
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Max gradient norm for the optimizer"
    )
    eval_steps: int = Field(
        default=200,
        description="Number of steps to evaluate the model"
    )
    save_steps: int = Field(
        default=200,
        description="Number of steps to save the model"
    )
    save_total_limit: int = Field(
        default=2,
        description="Number of total models to save"
    )
    
    output_dir: str = Field(
        default="outputs",
        description="Output directory for the model"
    )
    logging_steps: int = Field(
        default=10,
        description="Number of steps to log the model"
    )
    
    # Data processing parameters
    dataset_name: str = Field(
        default="younissk/tool-calling-mix",
        description="Dataset name for loading"
    )
    train_split: str = Field(
        default="train",
        description="Training split name"
    )
    eval_split: str = Field(
        default="validation", 
        description="Evaluation split name"
    )
    shuffle_seed: int = Field(
        default=42,
        description="Seed for shuffling data"
    )
    max_input_len: int = Field(
        default=2048,
        description="Maximum input sequence length"
    )
    max_label_len: int = Field(
        default=512,
        description="Maximum label sequence length"
    )
    
    base_model: str = Field(
        default="tiiuae/Falcon-H1-7B-Instruct",
        description="Base model for fine-tuning"
    )
    per_device_train_bs: int = Field(
        default=1,
        description="Per device training batch size"
    )
    per_device_eval_bs: int = Field(
        default=1,
        description="Per device evaluation batch size"
    )
    grad_accum: int = Field(
        default=4,
        description="Gradient accumulation steps"
    )
    difficulty_weights: bool = Field(
        default=False,
        description="Whether to use difficulty-based weighting"
    )
    eval_samples: int = Field(
        default=100,
        description="Number of samples for evaluation"
    )
    max_new_tokens_eval: int = Field(
        default=256,
        description="Maximum new tokens for evaluation"
    )
    wise_ft_alpha: float = Field(
        default=1.0,
        description="WiSE-FT interpolation alpha"
    )

    # Caching parameters
    enable_cache: bool = Field(
        default=True,
        description="Whether to enable dataset caching"
    )
    cache_dir: str = Field(
        default=".cache/datasets",
        description="Directory for caching processed datasets"
    )

    # Attention / kernels / precision
    attn_implementation: str = Field(
        default="auto",
        description="Attention backend: auto|flash_attention_2|sdpa|eager"
    )
    allow_tf32: bool = Field(
        default=True,
        description="Allow TF32 matmuls on Ampere+ (A100)"
    )
    enable_torch_compile: bool = Field(
        default=False,
        description="Enable torch.compile for the model forward"
    )
    torch_compile_mode: str = Field(
        default="max-autotune",
        description="torch.compile mode: default|reduce-overhead|max-autotune"
    )

    # DataLoader performance
    dataloader_num_workers: int = Field(
        default=8,
        description="Number of DataLoader workers"
    )
    dataloader_pin_memory: bool = Field(
        default=True,
        description="Pin host memory for DataLoader"
    )
    dataloader_prefetch_factor: int = Field(
        default=2,
        description="Prefetch factor per worker"
    )
    dataloader_persistent_workers: bool = Field(
        default=True,
        description="Keep workers alive between epochs"
    )

    # Sequence packing
    enable_packing: bool = Field(
        default=True,
        description="Enable constant-length sequence packing for training set"
    )
    pack_block_size: int = Field(
        default=2048,
        description="Packed sequence length (tokens)"
    )
    pack_eos_at_sample_end: bool = Field(
        default=True,
        description="Ensure EOS is present between samples when packing"
    )

    # Logging / evaluation controls
    log_tokens_per_second: bool = Field(
        default=True,
        description="Log tokens/sec throughput during training"
    )
    early_stopping_patience: int = Field(
        default=3,
        description="Early stopping patience (eval steps)"
    )



config = TrainingConfig()

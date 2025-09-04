"""Configuration for the Falcon-Twig fine-tuning project."""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    model_id: str = Field(
        default="tiiuae/Falcon-H1-7B-Instruct",
        description="Hugging Face model ID to fine-tune"
    )
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
    
    # Training parameters
    base_model: str = Field(
        default="tiiuae/Falcon-H1-7B-Instructf",
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



config = TrainingConfig()

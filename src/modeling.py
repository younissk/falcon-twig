import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import List, Any, cast
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore

# Try to import bitsandbytes, fallback for macOS ARM64
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    from peft import prepare_model_for_kbit_training  # type: ignore
    BNB_AVAILABLE: bool = True
except ImportError:
    BNB_AVAILABLE = False  # type: ignore[reportConstantRedefinition]

def get_compute_dtype() -> Any:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16  # type: ignore

def load_tokenizer(base_model: str) -> Any:
    tok = AutoTokenizer.from_pretrained(base_model)  # type: ignore
    if tok.pad_token is None:  # type: ignore
        tok.pad_token = tok.eos_token  # type: ignore
    tok.padding_side = "right"  # type: ignore
    return tok  # type: ignore

def load_model_standard(base_model: str, attn_implementation: str = "auto", enable_torch_compile: bool = False, torch_compile_mode: str = "max-autotune") -> Any:
    """Load model in standard fp16/bf16 mode for faster training when memory allows.
    
    This function loads the model without 4-bit quantization, which can speed up training by ~30%
    when sufficient VRAM is available (recommended: 70GB+ for 7B models).
    """
    compute_dtype = get_compute_dtype()
    _AutoCausalLM: Any = cast(Any, AutoModelForCausalLM)
    resolved_attn = None if attn_implementation == "auto" else attn_implementation
    try:
        model = _AutoCausalLM.from_pretrained(  # type: ignore
            base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation=resolved_attn,
        )  # type: ignore
    except Exception:
        model = _AutoCausalLM.from_pretrained(  # type: ignore
            base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )  # type: ignore
    return model

def load_model_4bit(base_model: str, attn_implementation: str = "auto", enable_torch_compile: bool = False, torch_compile_mode: str = "max-autotune") -> Any:
    """Load model in 4-bit mode for memory savings.
    
    This function uses QLoRA 4-bit quantization to reduce memory usage by ~6GB,
    but may increase training time by ~30%. Use load_model_standard() when memory allows.
    """
    if BNB_AVAILABLE:
        bnb = BitsAndBytesConfig(  # type: ignore
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_compute_dtype(),
        )
        _AutoCausalLM: Any = cast(Any, AutoModelForCausalLM)
        resolved_attn = None if attn_implementation == "auto" else attn_implementation
        try:
            model = _AutoCausalLM.from_pretrained(  # type: ignore
                base_model,
                quantization_config=bnb,  # type: ignore
                device_map="auto",
                attn_implementation=resolved_attn,
            )  # type: ignore
        except Exception:
            model = _AutoCausalLM.from_pretrained(  # type: ignore
                base_model,
                quantization_config=bnb,  # type: ignore
                device_map="auto",
                attn_implementation="sdpa",
            )  # type: ignore
        model = prepare_model_for_kbit_training(model)  # type: ignore
    else:
        # Fallback for macOS ARM64 without bitsandbytes - use standard loading
        print("Warning: bitsandbytes not available, falling back to standard loading")
        model = load_model_standard(base_model, attn_implementation, enable_torch_compile, torch_compile_mode)
    
    model.config.use_cache = False  # type: ignore
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # type: ignore
    # Optional compile for forward pass
    if enable_torch_compile:
        try:
            model = torch.compile(model, mode=torch_compile_mode, fullgraph=False)  # type: ignore
        except Exception:
            pass
    return model  # type: ignore

def infer_lora_targets(model: Any) -> List[str]:
    candidates = [
        "self_attention.query_key_value","self_attention.dense",
        "mlp.dense_h_to_4h","mlp.dense_4h_to_h",
        "attention.query_key_value","attention.dense",
        "query_key_value","dense_h_to_4h","dense_4h_to_h",
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
    ]
    present: set[str] = set()
    for name, module in model.named_modules():  # type: ignore
        if isinstance(module, nn.Linear):  # type: ignore
            lname = name.lower()  # type: ignore
            for c in candidates:
                if c in lname: 
                    present.add(c)  # type: ignore
    if not present:
        for name, module in model.named_modules():  # type: ignore
            if isinstance(module, nn.Linear) and "transformer" in name.lower():  # type: ignore
                present.add(name.split(".")[-1].lower())  # type: ignore
    return sorted(present)  # type: ignore

def apply_lora(model: Any, targets: List[str], r: int = 16, alpha: int = 32, dropout: float = 0.05) -> Any:
    lcfg = LoraConfig(  # type: ignore
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lcfg)  # type: ignore
    model.print_trainable_parameters()  # type: ignore
    return model

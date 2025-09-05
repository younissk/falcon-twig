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
    
    Args:
        base_model: Model ID from HuggingFace hub
        attn_implementation: One of "auto", "flash_attention_2", "sdpa", "eager"
                           Note: flash_attention_2 requires Python < 3.12 and CUDA toolkit
        enable_torch_compile: Whether to compile model with torch.compile
        torch_compile_mode: Compilation mode for torch.compile
    """
    # Check if flash attention is actually available when requested
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # type: ignore
        except ImportError:
            print("Warning: flash-attn not available, falling back to SDPA")
            attn_implementation = "sdpa"
    compute_dtype = get_compute_dtype()
    _AutoCausalLM: Any = cast(Any, AutoModelForCausalLM)
    resolved_attn = None if attn_implementation == "auto" else attn_implementation
    try:
        model = _AutoCausalLM.from_pretrained(  # type: ignore
            base_model,
            dtype=compute_dtype,
            device_map="auto",
            attn_implementation=resolved_attn,
        )  # type: ignore
    except Exception:
        model = _AutoCausalLM.from_pretrained(  # type: ignore
            base_model,
            dtype=compute_dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )  # type: ignore

    # Recommended for training with gradient checkpointing
    try:
        model.config.use_cache = False  # type: ignore
    except Exception:
        pass

    # Optional compile for forward pass
    if enable_torch_compile:
        try:
            model = torch.compile(model, mode=torch_compile_mode, fullgraph=False)  # type: ignore
        except Exception:
            pass
    return model

def load_model_4bit(base_model: str, attn_implementation: str = "auto", enable_torch_compile: bool = False, torch_compile_mode: str = "max-autotune") -> Any:
    """Load model in 4-bit mode for memory savings.
    
    This function uses QLoRA 4-bit quantization to reduce memory usage by ~6GB,
    but may increase training time by ~30%. Use load_model_standard() when memory allows.
    
    If bitsandbytes is not available or fails to load, falls back to standard fp16/bf16 loading.
    """
    try:
        # Check if bitsandbytes is properly installed and importable
        from transformers import BitsAndBytesConfig  # type: ignore
        from peft import prepare_model_for_kbit_training  # type: ignore
        import bitsandbytes  # type: ignore # noqa: F401
        
        bnb_config = BitsAndBytesConfig(  # type: ignore
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_compute_dtype(),
        )
        
        _AutoCausalLM: Any = cast(Any, AutoModelForCausalLM)
        resolved_attn = None if attn_implementation == "auto" else attn_implementation
        try:
            model: Any = _AutoCausalLM.from_pretrained(  # type: ignore
                base_model,
                quantization_config=bnb_config,  # type: ignore
                device_map="auto",
                attn_implementation=resolved_attn,
            )
        except Exception:
            model: Any = _AutoCausalLM.from_pretrained(  # type: ignore
                base_model,
                quantization_config=bnb_config,  # type: ignore
                device_map="auto",
                attn_implementation="sdpa",
            )

        # Prepare for QLoRA training
        model = prepare_model_for_kbit_training(model)  # type: ignore

        # Disable KV cache during training (recommended)
        try:
            model.config.use_cache = False  # type: ignore
        except Exception:
            pass

        # Optional compile for forward pass
        if enable_torch_compile:
            try:
                model = torch.compile(model, mode=torch_compile_mode, fullgraph=False)  # type: ignore
            except Exception:
                pass

        print("Successfully loaded model in 4-bit mode")
        return model
        
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Warning: bitsandbytes not available ({str(e)}), falling back to standard loading")
        return load_model_standard(base_model, attn_implementation, enable_torch_compile, torch_compile_mode)
    except Exception as e:
        print(f"Warning: 4-bit quantization failed ({str(e)}), falling back to standard loading")
        return load_model_standard(base_model, attn_implementation, enable_torch_compile, torch_compile_mode)
    
    # Unreachable legacy path retained for clarity above
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
        # Fallback: target common projection names across modern LMs
        common = {"q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","gate_proj","up_proj","down_proj","proj","dense","fc"}
        for name, module in model.named_modules():  # type: ignore
            if isinstance(module, nn.Linear):  # type: ignore
                last = name.split(".")[-1].lower()
                if last in common or any(t in last for t in common):
                    present.add(last)  # type: ignore
    return sorted(present)  # type: ignore

def apply_lora(model: Any, targets: List[str], r: int = 16, alpha: int = 32, dropout: float = 0.05) -> Any:
    # If provided targets won't match anything, fall back to inferred ones
    def _likely_matches_any(mod: str, targets: List[str]) -> bool:
        return any(t in mod for t in targets)

    has_match = False
    try:
        for name, module in model.named_modules():  # type: ignore
            if isinstance(module, nn.Linear):  # type: ignore
                if _likely_matches_any(name.lower().split(".")[-1], [t.lower() for t in targets]):
                    has_match = True
                    break
    except Exception:
        pass

    chosen_targets = targets[:]
    if not has_match:
        auto = infer_lora_targets(model)
        if auto:
            print(f"[LoRA] Provided targets {targets} did not match; using inferred targets: {auto}")
            chosen_targets = auto
        else:
            # Last-resort: target all Linear layer names (deduped) except lm_head
            names: List[str] = []
            try:
                for name, module in model.named_modules():  # type: ignore
                    if isinstance(module, nn.Linear):  # type: ignore
                        last = name.split(".")[-1]
                        if last != "lm_head":
                            names.append(last)
            except Exception:
                pass
            dedup = sorted({n for n in names})
            print(f"[LoRA] Falling back to broad Linear targets: {dedup}")
            chosen_targets = dedup

    lcfg = LoraConfig(  # type: ignore
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=chosen_targets,
    )
    model = get_peft_model(model, lcfg)  # type: ignore
    try:
        model.print_trainable_parameters()  # type: ignore
    except Exception:
        pass
    return model

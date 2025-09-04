import json
from typing import Dict, Any, Tuple, List
from datasets import load_dataset  # type: ignore
from src.config import TrainingConfig

SYSTEM_FALLBACK = "You are a helpful assistant that can call tools when appropriate."

def _ensure_list(x: Any) -> List[Any]:
    if x is None: 
        return []
    return x if isinstance(x, list) else [x]  # type: ignore

def _normalize_tools_from_json(tools_json: Any) -> List[Dict[str, Any]]:
    if tools_json is None:
        return []
    raw = tools_json
    if isinstance(raw, str):
        try: 
            raw = json.loads(raw)
        except Exception: 
            raw = [{"name": raw.strip()}]
    tools = _ensure_list(raw)
    out: List[Dict[str, Any]] = []
    for t in tools:
        if isinstance(t, dict) and "function" in t and isinstance(t["function"], dict):
            t = t["function"] # type: ignore
        if isinstance(t, str):
            try: 
                t = json.loads(t)
            except Exception: 
                t = {"name": t}
        if not isinstance(t, dict):
            t = {"name": str(t)}
        out.append(t)  # type: ignore
    return out

def _normalize_messages_from_json(messages_json: Any) -> List[Dict[str, str]]:
    msgs = messages_json
    if isinstance(msgs, str):
        try: 
            msgs = json.loads(msgs)
        except Exception: 
            msgs = []
    if not isinstance(msgs, list):
        msgs = []
    fixed: List[Dict[str, str]] = []
    for m in msgs:  # type: ignore
        role = (m or {}).get("role", "user")  # type: ignore
        content = (m or {}).get("content", "")  # type: ignore
        if isinstance(content, list):
            content = " ".join([p.get("text","") if isinstance(p, dict) else str(p) for p in content])  # type: ignore
        fixed.append({"role": role, "content": str(content)})  # type: ignore
    return fixed

def _render_toolblock(tools: List[Dict[str, Any]]) -> str:
    if not tools: 
        return ""
    lines = ["Available tools:"]
    for t in tools:
        lines.append(f"- {t.get('name','')}: {t.get('description','')}".rstrip(": "))
    return "\n".join(lines)

def build_prompt(example: Dict[str, Any], tokenizer: Any, max_in: int, max_lbl: int) -> Dict[str, Any]:
    """Build a single-sequence causal LM example with masked prompt.

    Decoder-only models expect `input_ids` and `labels` to be the same
    length. We therefore concatenate the prompt and target into one
    sequence and set labels for the prompt tokens to -100 so they don't
    contribute to the loss.
    """
    tools = _normalize_tools_from_json(example.get("tools_json"))
    msgs  = _normalize_messages_from_json(example.get("messages_json"))

    sys_txt = SYSTEM_FALLBACK
    if msgs and msgs[0].get("role") == "system":
        sys_txt = msgs[0].get("content", SYSTEM_FALLBACK)
        msgs = msgs[1:]

    parts = [f"<|system|>\n{sys_txt}"]
    tb = _render_toolblock(tools)
    if tb:
        parts.append(tb)
    for m in msgs:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    prompt_text = "\n\n".join(parts).strip()

    tgt = example.get("target_json", "")
    if isinstance(tgt, str):
        try:
            tgt = json.dumps(json.loads(tgt), ensure_ascii=False)
        except Exception:
            pass
    elif isinstance(tgt, (dict, list)):
        tgt = json.dumps(tgt, ensure_ascii=False)
    else:
        tgt = str(tgt)

    # Tokenize prompt and target separately
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_in, add_special_tokens=True
    )["input_ids"]
    # Avoid duplicating special tokens; we'll optionally add eos below
    target_ids = tokenizer(
        tgt, truncation=True, max_length=max_lbl, add_special_tokens=False
    )["input_ids"]

    # Ensure we end the target with eos if available so the model learns to stop
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        if len(target_ids) == 0 or target_ids[-1] != eos_id:
            target_ids = target_ids + [eos_id]

    # Build final input and labels
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids[:]

    return {"input_ids": input_ids, "labels": labels}

def load_and_prepare(config: TrainingConfig, tokenizer: Any) -> Tuple[Any, Any]:
    ds = load_dataset(config.dataset_name)  # type: ignore
    train = ds[config.train_split]  # type: ignore
    valid = ds[config.eval_split]  # type: ignore

    train = train.shuffle(seed=config.shuffle_seed)  # type: ignore

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]: 
        return build_prompt(ex, tokenizer, config.max_input_len, config.max_label_len)
    train = train.map(_map_fn)  # type: ignore
    valid = valid.map(_map_fn)  # type: ignore
    return train, valid  # type: ignore

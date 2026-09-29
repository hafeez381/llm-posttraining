"""`--mode smoke|full`: smoke swaps every model for a tiny test model so the whole path runs on a laptop CPU."""
from __future__ import annotations

import copy

# Tiny Qwen2.5-architecture models that keep the real Qwen2.5 tokenizer and chat template,
# so prompt rendering and truncation behave exactly as in full runs.
TINY_CAUSAL_LM = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_SEQ_CLS = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"


def add_mode_arg(parser) -> None:
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)


def resolve_models(cfg: dict, mode: str) -> dict:
    """Return the config with model ids resolved for `mode`. Full mode returns it unchanged."""
    if mode == "full":
        return cfg
    if mode != "smoke":
        raise ValueError(mode)
    cfg = copy.deepcopy(cfg)
    cfg.update(
        base_model=TINY_CAUSAL_LM,
        ai_judge_model=TINY_CAUSAL_LM,
        reward_model=TINY_SEQ_CLS,
        reward_tokenizer=TINY_CAUSAL_LM,
        value_model_init=TINY_SEQ_CLS,
        dtype="float32",
        quantize_frozen_models=False,
    )
    cfg["mode"] = "smoke"
    return cfg


def adapter_kwargs(cfg: dict, adapter_path: str | None) -> dict:
    """`load_policy` kwargs for a supplied adapter. Its shapes don't fit the tiny model, so smoke uses a fresh LoRA."""
    if cfg.get("mode") == "smoke" and adapter_path:
        return {"adapter_path": None, "fresh_lora": True}
    return {"adapter_path": adapter_path}

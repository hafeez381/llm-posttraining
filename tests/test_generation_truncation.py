import torch
from transformers import AutoModelForCausalLM

from common.generation import batch_generate
from common.models import load_tokenizer
from common.runmode import TINY_CAUSAL_LM

HEADER = "<|im_start|>assistant\n"


def test_truncated_prompt_keeps_the_assistant_header():
    tok = load_tokenizer(TINY_CAUSAL_LM)
    model = AutoModelForCausalLM.from_pretrained(TINY_CAUSAL_LM)
    long_prompt = [{"role": "user", "content": " ".join(["word"] * 300)}]
    rendered = tok.apply_chat_template(long_prompt, tokenize=False, add_generation_prompt=True)
    assert len(tok(rendered)["input_ids"]) > 256

    torch.manual_seed(0)
    out = batch_generate(model, tok, [long_prompt], max_prompt_length=256, max_new_tokens=2)
    prompt_ids = out["sequences"][0, : out["prompt_width"]]
    assert len(prompt_ids) == 256
    assert tok.decode(prompt_ids).endswith(HEADER)
    assert tok.truncation_side == "right"  # restored for other callers

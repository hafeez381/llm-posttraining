from common.data import encode_prompt_response, prompt_messages, read_jsonl
from common.metrics import parse_word_limit
from common.models import load_tokenizer
from common.runmode import TINY_CAUSAL_LM


def test_word_limit_parsed_for_every_fixed_prompt():
    rows = read_jsonl("data/word_limit_prompts.jsonl")
    limits = [parse_word_limit(r["messages"][-1]["content"]) for r in rows]
    assert limits == [40, 25, 30, 35, 45, 30, 35, 40, 30, 45]


def test_encoded_prompts_are_real_prompts():
    """Under transformers 5.x, apply_chat_template(tokenize=True) returns a BatchEncoding and every prompt
    silently encodes to 2 tokens; the pinned 4.57.1 returns token ids."""
    tok = load_tokenizer(TINY_CAUSAL_LM)
    for row in read_jsonl("data/gsm8k_eval.jsonl")[:5]:
        ids, rmask = encode_prompt_response(tok, prompt_messages(row), "The answer is 4.", max_length=768)
        assert len(ids) - sum(rmask) > 50

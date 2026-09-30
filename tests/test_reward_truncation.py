import torch

from common.generation import score_reward_pairs
from common.models import load_tokenizer
from common.runmode import TINY_CAUSAL_LM


class LastTokenProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = None

    def forward(self, input_ids, attention_mask):
        self.seen = input_ids
        return type("Out", (), {"logits": torch.zeros(input_ids.shape[0], 1)})()


def test_reward_model_sees_the_end_of_a_long_response():
    tok = load_tokenizer(TINY_CAUSAL_LM)
    probe = LastTokenProbe()
    prompt = [{"role": "user", "content": " ".join(["context"] * 400)}]
    score_reward_pairs(probe, tok, [prompt], ["the answer is FINAL"], max_length=128)
    assert probe.seen.shape[1] == 128
    assert tok.decode(probe.seen[0]).rstrip().endswith("the answer is FINAL<|im_end|>")
    assert tok.truncation_side == "right"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common.generation import response_token_logprobs
from common.runmode import TINY_CAUSAL_LM


def test_logprobs_use_the_sampling_temperature():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY_CAUSAL_LM).eval()
    seq = torch.randint(0, 1000, (2, 12))
    attn = torch.ones_like(seq)
    width = 8
    response = seq[:, width:]
    with torch.no_grad():
        logits = model(input_ids=seq, attention_mask=attn).logits[:, width - 1 : -1].float()
        for t in (1.0, 0.5):
            logp, _ = response_token_logprobs(model, seq, attn, width, response, temperature=t)
            expected = F.log_softmax(logits / t, -1).gather(-1, response.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(logp, expected)
        default, _ = response_token_logprobs(model, seq, attn, width, response)
        torch.testing.assert_close(default, F.log_softmax(logits, -1).gather(-1, response.unsqueeze(-1)).squeeze(-1))

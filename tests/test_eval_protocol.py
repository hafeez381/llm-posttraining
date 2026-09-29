import math

import torch

from common.eval_protocol import summarize
from common.metrics import sampled_kl
from common.rl_utils import approx_kl_k3, exact_entropy, seq_kl


def test_seq_kl_sums_masked_tokens():
    p = torch.tensor([[-1.0, -2.0, -3.0]])
    r = torch.tensor([[-1.5, -1.0, -9.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])
    assert seq_kl(p, r, m).item() == 0.5 - 1.0


def test_k3_hand_computed():
    new, old = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])
    r = math.log(0.25) - math.log(0.5)
    assert math.isclose(approx_kl_k3(new, old, torch.ones(1)).item(), math.exp(r) - r - 1, rel_tol=1e-6)
    assert approx_kl_k3(new, new, torch.ones(1)).item() == 0.0


def test_exact_entropy_uniform_and_masked():
    logits = torch.zeros(1, 3, 4)  # uniform over 4 tokens: entropy log 4 per token
    m = torch.tensor([[1.0, 1.0, 0.0]])
    assert math.isclose(exact_entropy(logits, m).item(), 2 * math.log(4), rel_tol=1e-6)


def test_pooled_token_kl_equals_the_course_helper():
    torch.manual_seed(0)
    p, r = torch.randn(3, 5), torch.randn(3, 5)
    m = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.float32)
    kl = seq_kl(p, r, m)
    rows = [{"kl_seq": float(kl[i]), "n_tokens": int(m[i].sum()), "rm_score": 0.0, "entropy_sampled_seq": 0.0,
             "entropy_exact_seq": 0.0, "eos": True, "truncated": False} for i in range(3)]
    s = summarize(rows)
    assert math.isclose(s["kl_tokmean"], sampled_kl(p, r, m).item(), rel_tol=1e-5)
    assert math.isclose(s["kl_seq_mean"], kl.mean().item(), rel_tol=1e-5)
    assert "kl_convention" in s

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def logprobs_eval_mode(model):
    """Run a log-prob forward with dropout off, restoring the previous mode afterwards.

    Gradients still flow; only dropout changes. With LoRA dropout active, two forwards with identical
    weights give different log-probs, so the PPO/GRPO ratio would differ from 1 before any update.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()



def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model

def seq_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence sampled KL estimate: sum over response tokens of log pi - log pi_ref (the k1 estimator)."""
    return ((policy_logp - ref_logp) * mask).sum(-1)


def approx_kl_k3(logp_new: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimate of KL(new || old) on samples from new: exp(r) - r - 1 with r = log old - log new.

    Non-negative and unbiased; the form used inside the GRPO loss.
    """
    r = logp_old - logp_new
    return (torch.exp(r) - r - 1.0) * mask


def exact_entropy(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-sequence sum of full-distribution token entropies, -sum_v p log p, over response tokens."""
    logp = F.log_softmax(logits.float() / temperature, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * mask).sum(-1)

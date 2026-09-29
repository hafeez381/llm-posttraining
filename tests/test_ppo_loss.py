import math

import torch

from task2_ppo.ppo import ppo_policy_loss


def test_ppo_clip_is_pessimistic_min():
    # rho = 1.5, eps = 0.2. A > 0: the clipped term 1.2 is the smaller one, so the loss is -1.2.
    new, old = torch.tensor([[math.log(1.5)]]), torch.zeros(1, 1)
    mask = torch.ones(1, 1)
    loss, _, _ = ppo_policy_loss(new, old, torch.tensor([[1.0]]), mask, eps=0.2)
    assert math.isclose(loss.item(), -1.2, rel_tol=1e-6)
    # A < 0: the unclipped term -1.5 is the smaller one, so the full penalty applies: loss +1.5.
    loss, _, _ = ppo_policy_loss(new, old, torch.tensor([[-1.0]]), mask, eps=0.2)
    assert math.isclose(loss.item(), 1.5, rel_tol=1e-6)


def test_clipped_tokens_get_no_gradient_only_when_improving():
    new = torch.tensor([[math.log(1.5), math.log(1.5)]], requires_grad=True)
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 2), torch.tensor([[1.0, -1.0]]), torch.ones(1, 2), eps=0.2)
    loss.backward()
    assert new.grad[0, 0].item() == 0.0 and new.grad[0, 1].item() != 0.0

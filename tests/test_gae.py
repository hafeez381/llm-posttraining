import math

import torch

from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


def test_gae_hand_computed():
    # One sequence, 3 valid steps + 1 pad, reward only at the end.
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.5, 0.6, 0.7, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    d2 = 1.0 - 0.7                     # last valid step bootstraps with 0
    d1 = 0.0 + 0.7 - 0.6
    d0 = 0.0 + 0.6 - 0.5
    a2, a1 = d2, d1 + 0.95 * d2
    a0 = d0 + 0.95 * a1
    torch.testing.assert_close(adv, torch.tensor([[a0, a1, a2, 0.0]]))
    torch.testing.assert_close(ret[:, :3], adv[:, :3] + values[:, :3])


def test_shaped_rewards_terminal_position():
    logp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    ref = torch.tensor([[-1.5, -1.0, -3.0, -9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    r = shaped_rewards(torch.tensor([2.0]), logp, ref, mask, beta_kl=0.1)
    torch.testing.assert_close(r, torch.tensor([[-0.05, 0.1, 2.0, 0.0]]))


def test_whiten_masked():
    adv = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    w = normalize_advantages(adv, mask)
    valid = w[mask.bool()]
    assert math.isclose(valid.mean().item(), 0.0, abs_tol=1e-6)
    assert math.isclose(valid.std(unbiased=False).item(), 1.0, rel_tol=1e-5)
    assert w[0, 3].item() == 0.0


def test_shaped_rewards_without_kl_are_terminal_only():
    logp = torch.tensor([[-1.0, -2.0, -3.0]])
    ref = torch.tensor([[-1.5, -1.0, -3.5]])
    r = shaped_rewards(torch.tensor([2.0]), logp, ref, torch.ones(1, 3), beta_kl=0.0)
    torch.testing.assert_close(r, torch.tensor([[0.0, 0.0, 2.0]]))

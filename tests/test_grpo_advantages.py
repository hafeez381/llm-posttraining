import torch

from task3_grpo.grpo import group_relative_advantages


def test_grpo_advantages_are_per_group():
    rewards = torch.tensor([0.0, 1.0, 10.0, 11.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    adv = group_relative_advantages(rewards, groups)
    torch.testing.assert_close(adv, torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0]))


def test_zero_std_group_gets_zero_advantage():
    adv = group_relative_advantages(torch.tensor([3.0, 3.0, 3.0, 0.0, 2.0]), torch.tensor([7, 7, 7, 9, 9]))
    assert torch.equal(adv[:3], torch.zeros(3))


def test_groups_need_not_be_contiguous():
    rewards = torch.tensor([0.0, 10.0, 1.0, 11.0])
    adv = group_relative_advantages(rewards, torch.tensor([0, 1, 0, 1]))
    torch.testing.assert_close(adv, torch.tensor([-1.0, -1.0, 1.0, 1.0]))


def test_std_scaled_variance_is_one():
    torch.manual_seed(0)
    rewards = torch.randn(4 * 8)
    groups = torch.arange(4).repeat_interleave(8)
    adv = group_relative_advantages(rewards, groups)
    for g in range(4):
        a = adv[groups == g]
        assert abs(a.mean().item()) < 1e-5 and abs(a.var(unbiased=False).item() - 1.0) < 1e-5

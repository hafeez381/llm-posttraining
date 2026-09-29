import math

import torch

from task1_dpo.dpo import dpo_loss


def test_dpo_loss_is_log2_when_policy_equals_reference():
    logp = torch.tensor([-12.0, -30.5, -7.25])
    rej = torch.tensor([-9.0, -41.0, -8.0])
    loss, stats = dpo_loss(logp, rej, logp.clone(), rej.clone(), beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert stats["preference_accuracy"].item() == 0.0  # every margin is exactly 0; accuracy counts m > 0


def test_dpo_loss_hand_computed():
    # policy log-ratio chosen - rejected = 5, reference = 2, so m = 3 and the logit is 0.1 * 3.
    loss, stats = dpo_loss(torch.tensor([-10.0]), torch.tensor([-15.0]),
                           torch.tensor([-11.0]), torch.tensor([-13.0]), beta=0.1)
    assert math.isclose(loss.item(), -math.log(1 / (1 + math.exp(-0.3))), rel_tol=1e-6)
    assert math.isclose(stats["margin_mean"].item(), 3.0, rel_tol=1e-6)
    assert math.isclose(stats["chosen_reward"].item(), 0.1, rel_tol=1e-5)
    assert math.isclose(stats["rejected_reward"].item(), -0.2, rel_tol=1e-5)


def test_dpo_accuracy_uses_reference():
    # The policy prefers chosen by 1 nat, but the reference preferred it by 4: the learned change is -3.
    _, stats = dpo_loss(torch.tensor([-10.0]), torch.tensor([-11.0]),
                        torch.tensor([-10.0]), torch.tensor([-14.0]), beta=0.1)
    assert stats["preference_accuracy"].item() == 0.0


def test_one_step_raises_chosen_and_lowers_rejected():
    chosen = torch.tensor([-10.0], requires_grad=True)
    rejected = torch.tensor([-12.0], requires_grad=True)
    loss, _ = dpo_loss(chosen, rejected, torch.tensor([-10.0]), torch.tensor([-12.0]), beta=0.1)
    loss.backward()
    with torch.no_grad():
        new_chosen, new_rejected = chosen - 1.0 * chosen.grad, rejected - 1.0 * rejected.grad
    assert new_chosen.item() > chosen.item() and new_rejected.item() < rejected.item()

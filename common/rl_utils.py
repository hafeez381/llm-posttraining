from __future__ import annotations

from contextlib import contextmanager


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


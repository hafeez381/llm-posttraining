import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from common.data import load_yaml
from common.generation import response_token_logprobs
from common.models import make_lora_config
from common.rl_utils import logprobs_eval_mode
from common.runmode import TINY_CAUSAL_LM


def _lora_model_in_train_mode():
    torch.manual_seed(0)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY_CAUSAL_LM), make_lora_config(load_yaml("configs/base.yaml")))
    # Non-zero LoRA B so dropout on the LoRA input actually changes the output.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=1.0)
    return model.train()


def _logp(model, seq):
    return response_token_logprobs(model, seq, torch.ones_like(seq), 8, seq[:, 8:])[0]


def test_dropout_makes_train_mode_logprobs_noisy():
    """Why the fix is needed: in train mode, identical weights give ratio != 1."""
    model = _lora_model_in_train_mode()
    seq = torch.randint(0, 1000, (2, 16))
    assert not torch.equal(_logp(model, seq), _logp(model, seq))


def test_eval_mode_logprobs_are_exact_and_still_differentiable():
    model = _lora_model_in_train_mode()
    seq = torch.randint(0, 1000, (2, 16))
    with logprobs_eval_mode(model):
        old = _logp(model, seq).detach()
        new = _logp(model, seq)
    assert model.training  # previous mode restored
    ratio = torch.exp(new - old)
    assert torch.equal(ratio, torch.ones_like(ratio))
    new.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)


def test_disable_dropout_survives_train_mode_and_checkpointing():
    from common.rl_utils import disable_dropout
    model = disable_dropout(_lora_model_in_train_mode())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    model.train()
    seq = torch.randint(0, 1000, (2, 16))
    first = _logp(model, seq)
    assert torch.equal(first, _logp(model, seq))
    first.sum().backward()

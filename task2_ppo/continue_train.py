from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.guards import assert_fixed_files, assert_same_midpoint, assert_train_split, sha256_file
from common.logging_utils import append_jsonl, set_seed
from common.metrics import masked_mean, sample_entropy
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from common.results import finalize, reconcile, start_run
from common.rl_utils import approx_kl_k3, disable_dropout, exact_entropy, logprobs_eval_mode, seq_kl
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.evaluate import logprob_temperature
from task2_ppo.ppo import (
    clip_affected_fraction,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)

RATIO_TOLERANCE = 1e-3


def value_checkpoint(cfg: dict) -> str:
    if cfg.get("mode") == "smoke":
        return snapshot_download(cfg["value_model_init"])
    return cfg["paths"]["ppo_midpoint_value"]


def response_values(value_model, sequences, attention_mask, prompt_width: int, n: int, alignment: str):
    base = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(base, base.base_model_prefix)
    hidden = backbone(input_ids=sequences, attention_mask=attention_mask, output_hidden_states=True,
                      return_dict=True, use_cache=False).hidden_states[-1]
    start = {"prev": prompt_width - 1, "same": prompt_width}[alignment]
    hidden = hidden[:, start:start + n]
    return value_model.score(hidden.float()).squeeze(-1).float()


def load_critic(cfg: dict):
    value_model = disable_dropout(load_value_model(cfg, value_checkpoint(cfg),
                                                   train_mode=cfg.get("value_train_mode", "head_only")))
    for p in trainable_parameters(value_model):
        p.data = p.data.float()
    return value_model


def prompt_sequence(cfg: dict, n: int) -> list[dict]:
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(rows))
    return [rows[i] for i in order[:n]]


def prepare_ppo_continuation(cfg: dict):
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = disable_dropout(load_policy(
        cfg,
        trainable=True,
        **adapter_kwargs(cfg, cfg["paths"]["ppo_midpoint_policy"]),
    ))
    value_model = load_critic(cfg)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def explained_variance(values, returns, mask) -> float:
    v, r = values[mask.bool()], returns[mask.bool()]
    var = r.var(unbiased=False)
    return float(1 - (r - v).var(unbiased=False) / var) if var > 0 else float("nan")


def ppo_update(b: dict, cfg: dict, row: dict, temperature: float) -> tuple[dict, dict]:
    tok, policy, value_model = b["tokenizer"], b["policy"], b["value_model"]
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    messages = prompt_messages(row)
    out = batch_generate(policy, tok, [messages], int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
                         temperature=cfg["generation"]["temperature"], top_p=cfg["generation"]["top_p"],
                         do_sample=cfg["generation"]["do_sample"])
    seq, attn, width = out["sequences"].clone(), out["attention_mask"], out["prompt_width"]
    resp = seq[:, width:]
    mask = out["response_mask"]
    args = (seq, attn, width, resp)

    with torch.no_grad():
        with logprobs_eval_mode(policy):
            old_logp, logits = response_token_logprobs(policy, *args, temperature=temperature)
            ent_exact = float(exact_entropy(logits, mask, temperature).sum() / mask.sum())
            del logits
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, *args, temperature=temperature)
        with logprobs_eval_mode(value_model):
            values = response_values(value_model, seq, attn, width, resp.shape[1], cfg["value_alignment"]) * mask
        reward_raw = float(score_reward_pairs(b["reward_model"], b["reward_tokenizer"], [messages], out["responses"],
                                              max_length=int(cfg["reward_max_length"]))[0])
        reward_eff = reward_raw - (0.0 if out["terminated_with_eos"][0] else float(cfg["missing_eos_penalty"]))
        rewards = shaped_rewards(torch.tensor([reward_eff], device=seq.device), old_logp, ref_logp, mask, beta)
        advantages, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        if not torch.allclose(returns, advantages + values) or not torch.isfinite(returns).all():
            raise SystemExit("BREACH: returns are not advantages + values before whitening, or are not finite")
        advantages = normalize_advantages(advantages, mask)

    stats = {k: [] for k in ("policy_loss", "value_loss", "clip_frac", "affected_frac", "grad_norm_policy",
                             "grad_norm_value")}
    for epoch in range(int(cfg["ppo_epochs"])):
        new_logp, _ = response_token_logprobs(policy, *args, temperature=temperature)
        loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, advantages, mask, eps)
        if epoch == 0:
            ratio_mean0 = float(masked_mean(ratio, mask))
        b["policy_optimizer"].zero_grad(set_to_none=True)
        loss.backward()
        gn_p = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
        b["policy_optimizer"].step()

        v_loss = value_mse_loss(response_values(value_model, seq, attn, width, resp.shape[1], cfg["value_alignment"]),
                                returns, mask)
        if not torch.isfinite(v_loss):
            raise SystemExit("BREACH: value loss is not finite")
        b["value_optimizer"].zero_grad(set_to_none=True)
        v_loss.backward()
        gn_v = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), float(cfg["max_grad_norm"]))
        b["value_optimizer"].step()
        for k, v in (("policy_loss", loss), ("value_loss", v_loss), ("clip_frac", clip_frac),
                     ("affected_frac", clip_affected_fraction(ratio, advantages, mask, eps)),
                     ("grad_norm_policy", gn_p), ("grad_norm_value", gn_v)):
            stats[k].append(v.item() if torch.is_tensor(v) else float(v))

    with torch.no_grad(), logprobs_eval_mode(policy):
        post_logp, _ = response_token_logprobs(policy, *args, temperature=temperature)
    n = int(mask.sum())
    record = {
        "prompt_id": row["prompt_id"],
        "reward_raw": reward_raw, "reward_eff": reward_eff,
        "kl_seq": float(seq_kl(old_logp, ref_logp, mask)[0]), "kl_tok": float(seq_kl(old_logp, ref_logp, mask)[0]) / n,
        "entropy_sampled": float(sample_entropy(old_logp, mask)), "entropy_exact": ent_exact,
        **{k: float(np.mean(v)) for k, v in stats.items() if not k.startswith("grad_norm")},
        "grad_norm_policy": max(stats["grad_norm_policy"]), "grad_norm_value": max(stats["grad_norm_value"]),
        "grad_clipped_policy": max(stats["grad_norm_policy"]) > float(cfg["max_grad_norm"]),
        "approx_kl_post": float(masked_mean(approx_kl_k3(old_logp, post_logp, mask), mask)),
        "resp_len": n, "eos": bool(out["terminated_with_eos"][0]), "truncated": bool(out["truncated"][0]),
        "ratio_mean_epoch0": ratio_mean0,
        "value_explained_var": explained_variance(values, returns, mask),
        "per_epoch": stats,
    }
    rollout = {"prompt_id": row["prompt_id"], "response": out["responses"][0], "reward_raw": reward_raw,
               "reward_eff": reward_eff, "resp_len": n, "eos": record["eos"], "truncated": record["truncated"]}
    return record, rollout


def summarize_updates(rows: list[dict]) -> dict:
    col = lambda k: [r[k] for r in rows]
    return {
        "n_updates": len(rows),
        "prompt_ids": col("prompt_id"),
        "reward_eff_mean": float(np.mean(col("reward_eff"))),
        "reward_eff_first5_mean": float(np.mean(col("reward_eff")[:5])),
        "reward_eff_last5_mean": float(np.mean(col("reward_eff")[-5:])),
        "kl_seq_mean": float(np.mean(col("kl_seq"))),
        "kl_seq_last": rows[-1]["kl_seq"],
        "approx_kl_post_mean": float(np.mean(col("approx_kl_post"))),
        "approx_kl_post_max": float(np.max(col("approx_kl_post"))),
        "grad_norm_policy_max": float(np.max(col("grad_norm_policy"))),
        "grad_clipped_policy_frac": float(np.mean(col("grad_clipped_policy"))),
        "clip_frac_mean": float(np.mean(col("clip_frac"))),
        "affected_frac_mean": float(np.mean(col("affected_frac"))),
        "value_explained_var_mean": float(np.nanmean(col("value_explained_var"))),
        "ratio_mean_epoch0_max_dev": float(np.max(np.abs(np.array(col("ratio_mean_epoch0")) - 1))),
        "generated_tokens": int(np.sum(col("resp_len"))),
        "eos_rate": float(np.mean(col("eos"))),
    }


def _save_snapshot(b, state, snap_dir):
    tmp = snap_dir.with_name(snap_dir.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    b["policy"].save_pretrained(tmp / "policy")
    b["value_model"].save_pretrained(tmp / "value")
    torch.save({"policy": b["policy_optimizer"].state_dict(), "value": b["value_optimizer"].state_dict()},
               tmp / "optimizers.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(snap_dir, ignore_errors=True)
    tmp.rename(snap_dir)


def _load_snapshot(b, snap_dir) -> dict:
    set_peft_model_state_dict(b["policy"], load_file(snap_dir / "policy" / "adapter_model.safetensors"))
    set_peft_model_state_dict(b["value_model"], load_file(snap_dir / "value" / "adapter_model.safetensors"))
    opt = torch.load(snap_dir / "optimizers.pt")
    b["policy_optimizer"].load_state_dict(opt["policy"])
    b["value_optimizer"].load_state_dict(opt["value"])
    return json.loads((snap_dir / "state.json").read_text())


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard", mode: str = "full", resume: bool = False):
    cfg = resolve_models(load_yaml(config_path), mode)
    smoke = mode == "smoke"
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    if smoke:
        cfg.update(updates=min(int(cfg["updates"]), 2), max_response_length=16, snapshot_every_updates=1)
    assert_fixed_files(mode)
    if not smoke:
        assert_same_midpoint("ppo", cfg["paths"]["ppo_midpoint_policy"], cfg["paths"]["ppo_midpoint_value"])
    temperature = logprob_temperature(cfg, mode)
    prompts = prompt_sequence(cfg, int(cfg["updates"]))
    assert_train_split([r["prompt_id"] for r in prompts], eval_path=cfg["paths"]["rl_prompt_eval"])

    prefix = "_smoke/" if smoke else ""
    out = repo_path(output or f"outputs/{prefix}task2_ppo/{run_name}")
    snap_dir = out.with_name(out.name + "_snapshot")
    b = prepare_ppo_continuation(cfg)
    state = {"update": 0, "train_seconds": 0.0}
    if resume:
        state = _load_snapshot(b, snap_dir)
    run = start_run(f"results/{prefix}task2_ppo/{run_name}/train", mode=mode, cfg=cfg, seed=int(cfg["seed"]),
                    resume=resume)
    log_path, roll_path = run["outdir"] / "updates.jsonl", run["outdir"] / "rollouts.jsonl"
    if resume:
        for p in (log_path, roll_path):
            kept = [r for r in read_jsonl(p) if r["update"] <= state["update"]]
            p.write_text("".join(json.dumps(r) + "\n" for r in kept))

    for u in range(state["update"], int(cfg["updates"])):
        t0 = time.perf_counter()
        set_seed(int(cfg["seed"]) + u)
        record, rollout = ppo_update(b, cfg, prompts[u], temperature)
        if abs(record["ratio_mean_epoch0"] - 1) > RATIO_TOLERANCE:
            print(f"BREACH: update {u + 1} epoch-0 ratio mean {record['ratio_mean_epoch0']:.6f} is not within "
                  f"{RATIO_TOLERANCE} of 1; old log-probs and the training forward disagree")
            sys.exit(1)
        state["update"] = u + 1
        state["train_seconds"] += time.perf_counter() - t0
        append_jsonl(log_path, {"update": u + 1, **record})
        append_jsonl(roll_path, {"update": u + 1, **rollout})
        print(f"update {u + 1}/{cfg['updates']} reward={record['reward_eff']:.3f} kl_seq={record['kl_seq']:.3f} "
              f"pl={record['policy_loss']:.4f} vl={record['value_loss']:.4f} clip={record['clip_frac']:.4f} "
              f"akl={record['approx_kl_post']:.2e} len={record['resp_len']}", flush=True)
        if state["update"] % int(cfg["snapshot_every_updates"]) == 0 or state["update"] == int(cfg["updates"]):
            _save_snapshot(b, state, snap_dir)

    b["policy"].save_pretrained(out)
    b["value_model"].save_pretrained(out / "value")
    rows = read_jsonl(log_path)
    summary = summarize_updates(rows)
    extra = {
        "run_name": run_name, "clip_epsilon": cfg["clip_epsilon"], "kl_beta": cfg["kl_beta"],
        "budget": f"{cfg['updates']} updates x {cfg['prompts_per_update']} prompt, cap {cfg['max_response_length']}",
        "adapter": str(out.relative_to(repo_path("."))),
        "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
        "logprob_temperature": temperature, "value_alignment": cfg["value_alignment"],
        "train_seconds": round(state["train_seconds"], 1),
        "peak_vram_gb_all_segments": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available() else None,
    }
    finalize(run, log_path, summary, summarize_updates, extra=extra)
    reconcile(run["outdir"], {"updates.jsonl", "rollouts.jsonl", "summary.json"})
    return {"summary": summary, **extra}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.mode,
                  args.resume)
    print(f"saved {out['adapter']} sha256={out['adapter_sha256']} budget: {out['budget']}")


if __name__ == "__main__":
    main()

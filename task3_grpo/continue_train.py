from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.guards import assert_fixed_files, assert_same_midpoint, assert_train_split, sha256_file
from common.logging_utils import append_jsonl, set_seed
from common.metrics import sample_entropy
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.results import finalize, reconcile, start_run
from common.rl_utils import disable_dropout, exact_entropy, logprobs_eval_mode, seq_kl
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.evaluate import logprob_temperature
from task2_ppo.continue_train import RATIO_TOLERANCE, prompt_sequence
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(cfg: dict):
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = disable_dropout(load_policy(
        cfg,
        trainable=True,
        **adapter_kwargs(cfg, cfg["paths"]["grpo_midpoint_policy"]),
    ))
    reward_model, reward_tokenizer = load_reward_model(cfg)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "optimizer": optimizer,
    }


def grpo_update(b: dict, cfg: dict, row: dict, temperature: float, loss_type: str) -> tuple[dict, dict]:
    tok, policy, k = b["tokenizer"], b["policy"], int(cfg["num_generations"])
    cap = int(cfg["max_completion_length"])
    messages = prompt_messages(row)
    out = batch_generate(policy, tok, [messages] * k, int(cfg["max_prompt_length"]), cap,
                         temperature=cfg["generation"]["temperature"], top_p=cfg["generation"]["top_p"],
                         do_sample=cfg["generation"]["do_sample"])
    seq, attn, width = out["sequences"].clone(), out["attention_mask"], out["prompt_width"]
    args = (seq, attn, width, seq[:, width:])
    mask = out["response_mask"]
    token_mask = mask_truncated_sequences(mask, out["truncated"]) if cfg["mask_truncated_completions"] else mask

    with torch.no_grad():
        rewards = score_reward_pairs(b["reward_model"], b["reward_tokenizer"], [messages] * k, out["responses"],
                                     max_length=int(cfg["reward_max_length"])).float()
        adv = group_relative_advantages(rewards, torch.zeros(k, dtype=torch.long, device=rewards.device))
        with logprobs_eval_mode(policy):
            old_logp, logits = response_token_logprobs(policy, *args, temperature=temperature)
            ent_exact = float(exact_entropy(logits, mask, temperature).sum() / mask.sum())
            del logits
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, *args, temperature=temperature)

    new_logp, _ = response_token_logprobs(policy, *args, temperature=temperature)
    loss, stats = grpo_policy_loss(new_logp, old_logp, adv.to(new_logp.device), token_mask, ref_logp,
                                   float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), loss_type, cap)
    b["optimizer"].zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
    b["optimizer"].step()

    lengths = [int(n) for n in mask.sum(-1).tolist()]
    trained = [int(n) for n in token_mask.sum(-1).tolist()]
    norm = [cap] * k if loss_type == "dr_grpo" else [max(t, 1) for t in trained]
    rstd = float(rewards.std(unbiased=False))
    record = {
        "prompt_id": row["prompt_id"],
        "reward_mean": float(rewards.mean()), "reward_std": rstd, "informative": rstd >= 1e-6,
        "n_masked": int(sum(t == 0 for t in trained)),
        "loss": loss.item(), "policy_term": float(stats["policy_term"]), "kl": float(stats["sampled_kl"]),
        "kl_seq": float(seq_kl(old_logp, ref_logp, mask).mean()),
        "grad_norm": float(grad_norm), "entropy_sampled": float(sample_entropy(old_logp, mask)),
        "entropy_exact": ent_exact, "len_mean": float(np.mean(lengths)),
        "clip_frac": float(stats["clip_fraction"]), "ratio_mean": float(stats["ratio_mean"]),
        "len_per_completion": lengths, "rewards": rewards.tolist(), "adv_per_completion": adv.tolist(),
        "weight_per_completion": [abs(a) / n if t else 0.0 for a, n, t in zip(adv.tolist(), norm, trained)],
    }
    rollout = {"prompt_id": row["prompt_id"], "responses": out["responses"], "rewards": rewards.tolist(),
               "lengths": lengths, "truncated": [bool(t) for t in out["truncated"]], "eos": out["terminated_with_eos"]}
    return record, rollout


def summarize_updates(rows: list[dict]) -> dict:
    col = lambda k: [r[k] for r in rows]
    return {
        "n_updates": len(rows),
        "prompt_ids": col("prompt_id"),
        "reward_mean": float(np.mean(col("reward_mean"))),
        "reward_first5_mean": float(np.mean(col("reward_mean")[:5])),
        "reward_last5_mean": float(np.mean(col("reward_mean")[-5:])),
        "reward_std_mean": float(np.mean(col("reward_std"))),
        "uninformative_frac": float(1 - np.mean(col("informative"))),
        "masked_completions": int(np.sum(col("n_masked"))),
        "kl_mean": float(np.mean(col("kl"))),
        "kl_seq_mean": float(np.mean(col("kl_seq"))),
        "grad_norm_max": float(np.max(col("grad_norm"))),
        "grad_clipped_frac": float(np.mean([g > 1.0 for g in col("grad_norm")])),
        "len_mean": float(np.mean(col("len_mean"))),
        "generated_tokens": int(sum(sum(r["len_per_completion"]) for r in rows)),
        "clip_frac_max": float(np.max(col("clip_frac"))),
    }


def _save_snapshot(b, state, snap_dir):
    tmp = snap_dir.with_name(snap_dir.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    b["policy"].save_pretrained(tmp)
    torch.save(b["optimizer"].state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(snap_dir, ignore_errors=True)
    tmp.rename(snap_dir)


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", mode: str = "full", resume: bool = False):
    cfg = resolve_models(load_yaml(config_path), mode)
    smoke = mode == "smoke"
    if updates is not None:
        cfg["updates"] = int(updates)
    if smoke:
        cfg.update(updates=min(int(cfg["updates"]), 2), max_completion_length=16, snapshot_every_updates=1)
    assert_fixed_files(mode)
    if not smoke:
        assert_same_midpoint("grpo", cfg["paths"]["grpo_midpoint_policy"])
    temperature = logprob_temperature(cfg, mode)
    prompts = prompt_sequence(cfg, int(cfg["updates"]))
    assert_train_split([r["prompt_id"] for r in prompts], eval_path=cfg["paths"]["rl_prompt_eval"])

    prefix = "_smoke/" if smoke else ""
    out = repo_path(output or f"outputs/{prefix}task3_grpo/{run_name}")
    snap_dir = out.with_name(out.name + "_snapshot")
    b = prepare_grpo_continuation(cfg)
    state = {"update": 0, "train_seconds": 0.0}
    if resume:
        set_peft_model_state_dict(b["policy"], load_file(snap_dir / "adapter_model.safetensors"))
        b["optimizer"].load_state_dict(torch.load(snap_dir / "optimizer.pt"))
        state = json.loads((snap_dir / "state.json").read_text())
    run = start_run(f"results/{prefix}task3_grpo/{run_name}/train", mode=mode, cfg=cfg, seed=int(cfg["seed"]),
                    resume=resume)
    log_path, roll_path = run["outdir"] / "updates.jsonl", run["outdir"] / "rollouts.jsonl"
    if resume:
        for p in (log_path, roll_path):
            kept = [r for r in read_jsonl(p) if r["update"] <= state["update"]]
            p.write_text("".join(json.dumps(r) + "\n" for r in kept))

    for u in range(state["update"], int(cfg["updates"])):
        t0 = time.perf_counter()
        set_seed(int(cfg["seed"]) + u)
        record, rollout = grpo_update(b, cfg, prompts[u], temperature, loss_type)
        if record["n_masked"] < int(cfg["num_generations"]) and (
                abs(record["ratio_mean"] - 1) > RATIO_TOLERANCE or record["clip_frac"] > 0):
            print(f"BREACH: update {u + 1} ratio mean {record['ratio_mean']:.6f}, clip fraction "
                  f"{record['clip_frac']:.4f}; with one epoch the ratio must be 1")
            sys.exit(1)
        state["update"] = u + 1
        state["train_seconds"] += time.perf_counter() - t0
        append_jsonl(log_path, {"update": u + 1, **record})
        append_jsonl(roll_path, {"update": u + 1, **rollout})
        print(f"update {u + 1}/{cfg['updates']} reward={record['reward_mean']:.3f}±{record['reward_std']:.3f} "
              f"kl={record['kl']:.2e} pt={record['policy_term']:.4f} gn={record['grad_norm']:.3f} "
              f"len={record['len_mean']:.0f} masked={record['n_masked']}", flush=True)
        if state["update"] % int(cfg["snapshot_every_updates"]) == 0 or state["update"] == int(cfg["updates"]):
            _save_snapshot(b, state, snap_dir)

    b["policy"].save_pretrained(out)
    rows = read_jsonl(log_path)
    summary = summarize_updates(rows)
    extra = {
        "run_name": run_name, "loss_type": loss_type,
        "budget": f"{cfg['updates']} updates x {cfg['prompts_per_update']} prompt x K={cfg['num_generations']}, "
                  f"cap {cfg['max_completion_length']}",
        "adapter": str(out.relative_to(repo_path("."))),
        "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
        "logprob_temperature": temperature, "train_seconds": round(state["train_seconds"], 1),
        "peak_vram_gb_all_segments": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available() else None,
    }
    finalize(run, log_path, summary, summarize_updates, extra=extra)
    reconcile(run["outdir"], {"updates.jsonl", "rollouts.jsonl", "summary.json"})
    return {"summary": summary, **extra}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.mode, args.resume)
    print(f"saved {out['adapter']} sha256={out['adapter_sha256']} budget: {out['budget']}")


if __name__ == "__main__":
    main()

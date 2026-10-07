from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.guards import assert_fixed_files, mark_analysis_only
from common.models import load_policy, load_tokenizer, trainable_parameters
from common.results import finalize, reconcile, start_run, write_rows
from common.rl_utils import disable_dropout, logprobs_eval_mode
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.evaluate import logprob_temperature
from task2_ppo.check_cache_logprobs import prompt_ids, response_ids
from task2_ppo.continue_train import RATIO_TOLERANCE
from task2_ppo.ppo import clip_affected_fraction, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

N_MINIBATCHES = 4


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def cache_advantages(rows: list[dict], cfg: dict) -> list[torch.Tensor]:
    n = max(int(r["response_tokens"]) for r in rows)
    pad = lambda key: torch.stack([torch.nn.functional.pad(r[key].float(), (0, n - len(r[key]))) for r in rows])
    mask = torch.stack([torch.arange(n) < int(r["response_tokens"]) for r in rows]).float()
    reward = torch.tensor([float(r["effective_terminal_reward"]) for r in rows])
    shaped = shaped_rewards(reward, pad("old_logprobs"), pad("ref_logprobs"), mask, float(cfg["kl_beta"]))
    adv, _ = compute_gae(shaped, pad("values") * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)
    return [adv[i, : int(r["response_tokens"])] for i, r in enumerate(rows)]


def score(ratio: torch.Tensor, adv: torch.Tensor, eps: float) -> dict:
    mask = torch.ones_like(ratio)
    loss, _, clip_frac = ppo_policy_loss(ratio.log(), torch.zeros_like(ratio), adv, mask, eps)
    l_cpi = float((ratio * adv).mean())
    return {"clip_frac": float(clip_frac), "affected_frac": float(clip_affected_fraction(ratio, adv, mask, eps)),
            "L_clip": float(-loss), "L_cpi": l_cpi, "L_cpi_minus_L_clip": l_cpi - float(-loss)}


def summarize(rows: list[dict]) -> dict:
    fields = sorted({r["field_eps"] for r in rows})
    by_field = {}
    for f in fields:
        sel = [r for r in rows if r["field_eps"] == f]
        by_field[f] = (torch.tensor([r["ratio"] for r in sel], dtype=torch.float64),
                       torch.tensor([r["adv"] for r in sel], dtype=torch.float64))
    reference = 0.2 if 0.2 in by_field else fields[len(fields) // 2]
    out = {"n_tokens_per_field": {str(f): len(by_field[f][0]) for f in fields}, "geometry_field_eps": reference,
           "geometry": {str(e): score(*by_field[reference], e) for e in fields},
           "dynamics": {str(f): score(*by_field[f], f) for f in fields}, "ratio_quantiles": {}}
    for f, (ratio, _) in by_field.items():
        q = np.quantile(ratio.numpy(), [0.01, 0.05, 0.5, 0.95, 0.99])
        out["ratio_quantiles"][str(f)] = {"q01": q[0], "q05": q[1], "q50": q[2], "q95": q[3], "q99": q[4],
                                          "max_abs_log_ratio": float(ratio.log().abs().max())}
    return out


def main():
    ap = argparse.ArgumentParser(description="Clipped surrogate, clip fraction and affected-token fraction on the "
                                             "cached PPO rollouts")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    temperature = logprob_temperature(cfg, args.mode)
    run = start_run(f"results/{'_smoke/' if smoke else ''}task2_ppo/clipping_cache", mode=args.mode, cfg=cfg,
                    seed=int(cfg["seed"]))

    cache = load_cached_rollouts(cfg["cached_rollouts"])[: 4 if smoke else None]
    messages = {r["prompt_id"]: prompt_messages(r) for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tok = load_tokenizer(cfg["base_model"])
    policy = mark_analysis_only(disable_dropout(load_policy(
        cfg, trainable=True, **adapter_kwargs(cfg, cfg["paths"]["ppo_midpoint_policy"]))))
    device = next(policy.parameters()).device
    midpoint = {k: v.detach().clone() for k, v in get_peft_model_state_dict(policy).items()}

    items = []
    for r in cache:
        p, y = prompt_ids(tok, messages[r["prompt_id"]], int(cfg["max_prompt_length"])), response_ids(tok, r)
        seq = torch.tensor([p + y], device=device)
        items.append({"args": (seq, torch.ones_like(seq), len(p), seq[:, len(p):]), "prompt_id": r["prompt_id"],
                      "long_prompt": len(tok.apply_chat_template(messages[r["prompt_id"]], tokenize=True,
                                                                 add_generation_prompt=True)) > int(cfg["max_prompt_length"])})
    for it, adv in zip(items, cache_advantages(cache, cfg)):
        it["adv"] = adv.to(device)

    def logprobs(it, grad=False):
        with torch.set_grad_enabled(grad):
            return response_token_logprobs(policy, *it["args"], temperature=temperature)[0][0]

    with logprobs_eval_mode(policy):
        for it in items:
            it["old"] = logprobs(it).detach()

    size = len(items) // N_MINIBATCHES
    rows, start_ratio = [], {}
    for eps in [float(e) for e in cfg["clip_values"]]:
        set_peft_model_state_dict(policy, midpoint)
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        first = []
        for m in range(N_MINIBATCHES):
            mb = items[m * size:(m + 1) * size]
            n_tok = sum(len(it["old"]) for it in mb)
            opt.zero_grad(set_to_none=True)
            for it in mb:
                new = logprobs(it, grad=True)
                if m == 0:
                    first.append((new.detach() - it["old"]).exp())
                loss, _, _ = ppo_policy_loss(new, it["old"], it["adv"], torch.ones_like(new), eps)
                (loss * len(new) / n_tok).backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            opt.step()
        start_ratio[str(eps)] = float(torch.cat(first).mean())
        with logprobs_eval_mode(policy):
            for i, it in enumerate(items):
                ratio = (logprobs(it).detach() - it["old"]).exp().cpu()
                rows += [{"field_eps": eps, "rollout": i, "prompt_id": it["prompt_id"], "t": t,
                          "ratio": float(ratio[t]), "adv": float(it["adv"][t]), "long_prompt": it["long_prompt"]}
                         for t in range(len(ratio))]

    rows_path = run["outdir"] / "per_token.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    n_cached = sum(int(r["response_tokens"]) for r in cache)
    extra = {"field_procedure": f"one epoch from the midpoint at the field's epsilon, {N_MINIBATCHES} minibatches of "
                                f"{size} rollouts", "kl_beta": cfg["kl_beta"], "start_ratio_mean": start_ratio,
             "cached_response_tokens": n_cached, "long_prompt_rollouts": sum(it["long_prompt"] for it in items)}
    finalize(run, rows_path, summary, summarize, extra=extra)
    reconcile(run["outdir"], {"per_token.jsonl", "summary.json"})

    for e, s in summary["geometry"].items():
        d = summary["dynamics"][e]
        print(f"eps={e}: geometry clip={s['clip_frac']:.4f} affected={s['affected_frac']:.4f} "
              f"L_clip={s['L_clip']:.4f} gap={s['L_cpi_minus_L_clip']:.2e} | dynamics clip={d['clip_frac']:.4f} "
              f"affected={d['affected_frac']:.4f} gap={d['L_cpi_minus_L_clip']:.2e} | start ratio={start_ratio[e]:.6f}")
    geo = [summary["geometry"][str(e)] for e in sorted(float(x) for x in summary["geometry"])]
    verdicts = {
        "clip fraction strictly falls with epsilon on the common field":
            all(a["clip_frac"] > b["clip_frac"] for a, b in zip(geo, geo[1:])),
        "affected fraction <= clip fraction": all(s["affected_frac"] <= s["clip_frac"] for s in geo),
        "ratio is 1 before the first step": all(abs(v - 1) <= RATIO_TOLERANCE for v in start_ratio.values()),
        "every cached response token is scored": all(v == n_cached for v in summary["n_tokens_per_field"].values()),
    }
    for gate, ok in verdicts.items():
        print(f"{'PASS' if ok else 'BREACH'}: {gate}")
    if smoke:
        print("smoke: the tiny model's ratios barely move, so the verdicts are not enforced")
        return
    if not all(verdicts.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

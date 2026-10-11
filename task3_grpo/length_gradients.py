from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import response_token_logprobs
from common.guards import assert_fixed_files, mark_analysis_only
from common.models import load_policy, load_tokenizer, trainable_parameters
from common.results import finalize, reconcile, start_run, write_rows
from common.rl_utils import disable_dropout
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.evaluate import logprob_temperature
from task2_ppo.check_cache_logprobs import prompt_ids
from task3_grpo.analyze_group_size import load_k8_cache, regroup_equal_generation_budget
from task3_grpo.grpo import group_relative_advantages

K = 4
LINEAR_SLOPE = 0.5


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def loglog_slope(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) > 2 else float("nan")


def summarize(rows: list[dict]) -> dict:
    out = {}
    for subset, keep in (("all", lambda r: True), ("T<=512", lambda r: r["T"] <= 512)):
        for sign, s_keep in (("all", lambda r: True), ("A>0", lambda r: r["adv"] > 0), ("A<0", lambda r: r["adv"] < 0)):
            sel = [r for r in rows if r["adv"] != 0 and keep(r) and s_keep(r)]
            t = np.array([r["T"] for r in sel], dtype=float)
            out[f"{subset}|{sign}"] = {"n": len(sel), **{
                f"{m}_{k}": f(t, np.array([r[k] for r in sel])) for k in ("gnorm_raw", "gnorm_grpo", "gnorm_dr")
                for m, f in (("spearman", spearman), ("loglog_slope", loglog_slope))}}
    sel = [r for r in rows if r["adv"] != 0]
    q = np.quantile([r["T"] for r in sel], [0.25, 0.5, 0.75])
    bins = np.digitize([r["T"] for r in sel], q)
    out["length_quartiles"] = {str(b): {k: float(np.mean([r[k] for r, bb in zip(sel, bins) if bb == b]))
                                        for k in ("T", "gnorm_raw", "gnorm_grpo", "gnorm_dr")} for b in range(4)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-completion policy-gradient norm vs length on the fixed GRPO cache, "
                                             "under canonical and Dr. GRPO normalization")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    temperature = logprob_temperature(cfg, args.mode)
    run = start_run(f"results/{'_smoke/' if smoke else ''}task3_grpo/length_gradients", mode=args.mode, cfg=cfg,
                    seed=int(cfg["seed"]))

    by_prompt = load_k8_cache(cfg["group_cache"])
    if smoke:
        by_prompt = dict(list(by_prompt.items())[:1])
    messages = {r["prompt_id"]: prompt_messages(r) for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tok = load_tokenizer(cfg["base_model"])
    policy = mark_analysis_only(disable_dropout(load_policy(
        cfg, trainable=True, **adapter_kwargs(cfg, cfg["paths"]["grpo_midpoint_policy"]))))
    device = next(policy.parameters()).device
    params = trainable_parameters(policy)
    l_max = int(cfg["max_completion_length"])

    rows = []
    for group in regroup_equal_generation_budget(by_prompt, K):
        by_gen = {int(c["generation_index"]): c for c in by_prompt[group["source_index"]]}
        adv = group_relative_advantages(torch.tensor(group["rewards"]), torch.zeros(K, dtype=torch.long))
        for c, a in zip([by_gen[i] for i in group["gen_indices"]], adv.tolist()):
            ids = tok(c["completion"], add_special_tokens=False)["input_ids"]
            ids += [tok.eos_token_id] if c["terminated_with_eos"] else []
            p = prompt_ids(tok, messages[c["prompt_id"]], int(cfg["max_prompt_length"]))
            gnorm = 0.0
            if a != 0:
                seq = torch.tensor([p + ids], device=device)
                logp = response_token_logprobs(policy, seq, torch.ones_like(seq), len(p), seq[:, len(p):],
                                               temperature=temperature)[0]
                policy.zero_grad(set_to_none=True)
                (a * logp.sum()).backward()
                gnorm = float(torch.sqrt(sum((q.grad.float() ** 2).sum() for q in params if q.grad is not None)))
            t = len(ids)
            rows.append({"prompt_id": c["prompt_id"], "gen_idx": int(c["generation_index"]), "T": t,
                         "T_cache": int(c["completion_tokens"]), "clipped_at_max": bool(c["clipped_at_max"]),
                         "adv": a, "gnorm_raw": gnorm, "gnorm_grpo": gnorm / t, "gnorm_dr": gnorm / l_max})
    policy.zero_grad(set_to_none=True)

    rows_path = run["outdir"] / "length_gradients.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    extra = {"k": K, "l_max": l_max, "n_token_count_mismatch": sum(r["T"] != r["T_cache"] for r in rows),
             "n_clipped_at_max": sum(r["clipped_at_max"] for r in rows)}
    finalize(run, rows_path, summary, summarize, extra=extra)
    reconcile(run["outdir"], {"length_gradients.jsonl", "summary.json"})

    for key in ("all|all", "all|A>0", "all|A<0", "T<=512|all"):
        s = summary[key]
        print(f"{key}: n={s['n']} spearman(T, |g|) raw={s['spearman_gnorm_raw']:.3f} "
              f"grpo={s['spearman_gnorm_grpo']:.3f} dr={s['spearman_gnorm_dr']:.3f} | log-log slope "
              f"raw={s['loglog_slope_gnorm_raw']:.3f} grpo={s['loglog_slope_gnorm_grpo']:.3f}")
    print(f"rows={len(rows)} token-count mismatches={extra['n_token_count_mismatch']} "
          f"clipped_at_max={extra['n_clipped_at_max']}")
    slope = summary["all|all"]["loglog_slope_gnorm_grpo"]
    if smoke:
        print("smoke: tiny model, the verdict is not enforced")
        return
    if slope > LINEAR_SLOPE:
        print(f"BREACH: canonical per-completion gradient norm grows with length (log-log slope {slope:.3f} > "
              f"{LINEAR_SLOPE}); the 1/T normalizer or the mask is wrong")
        sys.exit(1)
    print(f"PASS: canonical per-completion gradient norm does not grow linearly with length (slope {slope:.3f})")


if __name__ == "__main__":
    main()

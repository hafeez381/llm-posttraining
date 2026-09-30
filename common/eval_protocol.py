"""Shared held-out evaluation for Tasks 1-3: sample once per prompt, score with the RM, measure KL to the reference.

Sampling uses the release rollout config, one seeded sample per prompt: the reported KL is the sampled-response
estimator E_{y~pi}[log pi(y) - log pi_ref(y)], which greedy outputs would not estimate. Each fixed-size batch is
seeded with seed + batch_index, so re-evaluating the same policy reproduces every row.

KL convention: `kl_seq_mean` (per-sequence sum over response tokens, averaged over prompts) is the headline;
`kl_tokmean` (pooled token mean, the course `sampled_kl` helper over all tokens) is reported beside it.
Token counts include the EOS token when one was generated.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.results import finalize, reconcile, start_run, write_rows
from common.rl_utils import exact_entropy, logprobs_eval_mode, seq_kl
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models

KL_CONVENTION = (
    "kl_seq_mean: per-sequence sum of log pi - log pi_ref over response tokens, mean over prompts; "
    "kl_tokmean: pooled token mean over all response tokens (course sampled_kl helper)"
)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 6304
LOGPROB_CHUNK = 4


@torch.no_grad()
def evaluate_policy(policy, tokenizer, rm, rm_tokenizer, prompts: list[dict], *, max_prompt_length: int,
                    max_new_tokens: int, batch_size: int, reward_max_length: int, logprob_temperature: float,
                    generation: dict, seed: int = 6304) -> list[dict]:
    """One row per prompt. `prompts` items carry `prompt_id` and `messages`; the reference is `policy`
    with its adapter disabled, so the adapter is the only difference between the two."""
    rows = []
    for b, start in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[start:start + batch_size]
        set_seed(seed + b)
        out = batch_generate(policy, tokenizer, [p["messages"] for p in batch], max_prompt_length, max_new_tokens,
                             temperature=generation["temperature"], top_p=generation["top_p"],
                             do_sample=generation["do_sample"])
        mask = out["response_mask"]
        kl, ent_sampled, ent_exact = [], [], []
        for c in range(0, len(batch), LOGPROB_CHUNK):
            sl = slice(c, c + LOGPROB_CHUNK)
            args = (out["sequences"][sl], out["attention_mask"][sl], out["prompt_width"], out["response_ids"][sl])
            with logprobs_eval_mode(policy):
                logp, logits = response_token_logprobs(policy, *args, temperature=logprob_temperature)
                ent_exact.append(exact_entropy(logits, mask[sl], logprob_temperature))
                del logits
                with reference_mode(policy):
                    ref_logp, _ = response_token_logprobs(policy, *args, temperature=logprob_temperature)
            kl.append(seq_kl(logp, ref_logp, mask[sl]))
            ent_sampled.append(-(logp * mask[sl]).sum(-1))
        kl, ent_sampled, ent_exact = torch.cat(kl), torch.cat(ent_sampled), torch.cat(ent_exact)
        # The RM sees the original, untruncated messages plus the response.
        rm_scores = score_reward_pairs(rm, rm_tokenizer, [p["messages"] for p in batch], out["responses"],
                                       max_length=reward_max_length)
        for i, p in enumerate(batch):
            rows.append({
                "prompt_id": str(p["prompt_id"]),
                "batch_index": b,
                "response": out["responses"][i],
                "n_tokens": int(out["response_lengths"][i]),
                "eos": bool(out["terminated_with_eos"][i]),
                "truncated": bool(out["truncated"][i]),
                "rm_score": float(rm_scores[i]),
                "kl_seq": float(kl[i]),
                "entropy_sampled_seq": float(ent_sampled[i]),
                "entropy_exact_seq": float(ent_exact[i]),
            })
    return rows


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    q25, q75 = np.percentile(x, [25, 75])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = x[rng.integers(0, len(x), size=(BOOTSTRAP_RESAMPLES, len(x)))].mean(1)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "boot_se": float(boot.std(ddof=1))}


def summarize(rows: list[dict]) -> dict:
    n_tok = sum(r["n_tokens"] for r in rows)
    return {
        "n": len(rows),
        "rm_score": _stats([r["rm_score"] for r in rows]),
        "length": _stats([r["n_tokens"] for r in rows]),
        "kl_seq": _stats([r["kl_seq"] for r in rows]),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in rows])),
        "kl_tokmean": sum(r["kl_seq"] for r in rows) / max(n_tok, 1),
        "entropy_sampled_tokmean": sum(r["entropy_sampled_seq"] for r in rows) / max(n_tok, 1),
        "entropy_exact_tokmean": sum(r["entropy_exact_seq"] for r in rows) / max(n_tok, 1),
        "eos_rate": float(np.mean([r["eos"] for r in rows])),
        "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
        "kl_convention": KL_CONVENTION,
    }


def main():
    """Self-check: evaluating the same policy twice on 4 held-out RL prompts must give identical rows."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    outdir = "results/_smoke/eval_protocol" if smoke else "results/eval_protocol_check"
    run = start_run(outdir, mode=args.mode, cfg=cfg, seed=cfg["seed"])

    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, **adapter_kwargs(cfg, cfg["paths"]["ppo_midpoint_policy"]))
    rm, rm_tok = load_reward_model(cfg)
    prompts = [{"prompt_id": r.get("prompt_id", r.get("source_index")), "messages": prompt_messages(r)}
               for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])[:args.n]]
    kwargs = dict(max_prompt_length=cfg["max_prompt_length"],
                  max_new_tokens=32 if smoke else cfg["eval_max_response_length"],
                  batch_size=2, reward_max_length=cfg["reward_max_length"], logprob_temperature=1.0,
                  generation=cfg["generation"], seed=cfg["seed"])

    first = evaluate_policy(policy, tok, rm, rm_tok, prompts, **kwargs)
    second = evaluate_policy(policy, tok, rm, rm_tok, prompts, **kwargs)
    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, first)
    summary = summarize(first)
    finalize(run, rows_path, summary, summarize)
    reconcile(run["outdir"], {"rows.jsonl", "summary.json"})

    print(f"kl_seq_mean={summary['kl_seq_mean']:.4f} kl_tokmean={summary['kl_tokmean']:.4f} "
          f"rm_mean={summary['rm_score']['mean']:.4f} len_mean={summary['length']['mean']:.1f}")
    if first != second:
        print("BREACH: re-evaluating the same policy changed the rows; seeding does not pin the samples")
        sys.exit(1)
    print(f"PASS: re-evaluating the same policy reproduced all {len(first)} rows")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path
from common.eval_protocol import LOGPROB_CHUNK, evaluate_policy, summarize as summarize_generations
from common.guards import assert_fixed_files, sha256_file
from common.models import load_policy, load_reward_model, load_tokenizer
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.train import make_collate, pair_logprobs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


@torch.no_grad()
def score_pairs(model, tokenizer, rows, beta: float, max_length: int, batch_size: int) -> list[dict]:
    collate = make_collate(tokenizer, max_length)
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        chosen, rejected, meta = collate(batch)
        pc, pr, rc, rr = [torch.cat(t) for t in zip(*(
            pair_logprobs(model, *[{k: v[c:c + LOGPROB_CHUNK] for k, v in d.items()} for d in (chosen, rejected)])
            for c in range(0, len(batch), LOGPROB_CHUNK)))]
        margin = (pc - pr) - (rc - rr)
        loss = -F.logsigmoid(beta * margin)
        for i, row in enumerate(batch):
            out.append({
                "prompt_id": row["prompt_id"], "beta": beta,
                "logp_c": float(pc[i]), "logp_r": float(pr[i]), "ref_c": float(rc[i]), "ref_r": float(rr[i]),
                "margin": float(margin[i]), "loss_at_beta": float(loss[i]),
                "truncated_c": meta["truncated_c"][i], "truncated_r": meta["truncated_r"][i],
                "score_gap": float(row["score_chosen"]) - float(row["score_rejected"]),
                **{k: row[k] for k in ("length_stratum", "length_difference", "chosen_tokens", "rejected_tokens")
                   if k in row},
            })
    return out


def summarize_pairs(rows: list[dict]) -> dict:
    margins = np.array([r["margin"] for r in rows])
    correct = int((margins > 0).sum())
    out = {
        "n": len(rows), "beta": rows[0]["beta"],
        "loss_mean": float(np.mean([r["loss_at_beta"] for r in rows])),
        "accuracy": correct / len(rows), "accuracy_wilson95": wilson(correct, len(rows)),
        "margin_mean": float(margins.mean()), "margin_sd": float(margins.std(ddof=1)) if len(rows) > 1 else 0.0,
        "n_truncated_pairs": sum(r["truncated_c"] or r["truncated_r"] for r in rows),
    }
    strata = sorted({r["length_stratum"] for r in rows if "length_stratum" in r})
    if strata:
        out["by_stratum"] = {}
        for s in strata:
            m = [r["margin"] for r in rows if r.get("length_stratum") == s]
            k = sum(x > 0 for x in m)
            out["by_stratum"][s] = {"n": len(m), "accuracy": k / len(m), "accuracy_wilson95": wilson(k, len(m)),
                                    "margin_mean": float(np.mean(m))}
    return out


def logprob_temperature(cfg: dict, mode: str) -> float:
    if mode == "smoke":
        return 1.0
    t = cfg.get("logprob_temperature")
    if t is None:
        raise SystemExit("logprob_temperature is unset: run python -m task2_ppo.check_cache_logprobs --mode full "
                         "and commit the adopted value in configs/base.yaml first")
    return float(t)


def load_eval_policy(cfg, adapter: str | None):
    return load_policy(cfg, **adapter_kwargs(cfg, adapter)) if adapter else load_policy(cfg)


def evaluate(config_path: str, adapter: str | None, name: str, beta: float | None = None, mode: str = "full",
             budget: str = "none (reference policy)") -> dict:
    cfg = resolve_models(load_yaml(config_path), mode)
    smoke = mode == "smoke"
    assert_fixed_files(mode)
    beta = float(cfg["beta"] if beta is None else beta)
    temperature = logprob_temperature(cfg, mode)
    prefix = "_smoke/" if smoke else ""
    run = start_run(f"results/{prefix}task1_dpo/{name}/eval", mode=mode, cfg=cfg, seed=int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_eval_policy(cfg, adapter)
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])[: 4 if smoke else None]
    batch_size = 2 if smoke else int(cfg["eval_batch_size"])

    pairs = score_pairs(policy, tokenizer, rows, beta, int(cfg["max_sequence_length"]), batch_size)
    pairs_path = run["outdir"] / "pairs.jsonl"
    write_rows(pairs_path, pairs)

    rm, rm_tok = load_reward_model(cfg)
    prompts = [{"prompt_id": r["prompt_id"], "messages": prompt_messages_from_preference(r)} for r in rows]
    max_prompt = int(cfg["eval_max_prompt_length"])
    n_prompt_truncated = sum(
        len(tokenizer.apply_chat_template(p["messages"], tokenize=True, add_generation_prompt=True)) > max_prompt
        for p in prompts)
    gens = evaluate_policy(policy, tokenizer, rm, rm_tok, prompts, max_prompt_length=max_prompt,
                           max_new_tokens=32 if smoke else int(cfg["max_generation_tokens"]), batch_size=batch_size,
                           reward_max_length=int(cfg["reward_max_length"]), logprob_temperature=temperature,
                           generation=cfg["generation"], seed=int(cfg["seed"]))
    gens_path = run["outdir"] / "generations.jsonl"
    write_rows(gens_path, gens)

    adapter_file = repo_path(adapter) / "adapter_model.safetensors" if adapter else None
    extra = {"name": name, "adapter": adapter, "budget": budget, "beta": beta, "logprob_temperature": temperature,
             "adapter_sha256": sha256_file(adapter_file) if adapter_file and adapter_file.exists() else None,
             f"n_prompts_truncated_at_{max_prompt}": n_prompt_truncated}
    pair_summary = summarize_pairs(pairs)
    gen_summary = summarize_generations(gens)
    finalize(run, pairs_path, pair_summary, summarize_pairs, name="pairs_summary.json", extra=extra)
    finalize(run, gens_path, gen_summary, summarize_generations, name="generations_summary.json", extra=extra)
    reconcile(run["outdir"], {"pairs.jsonl", "generations.jsonl", "pairs_summary.json", "generations_summary.json"})
    print(f"{name}: loss@{beta}={pair_summary['loss_mean']:.4f} acc={pair_summary['accuracy']:.4f} "
          f"margin={pair_summary['margin_mean']:.3f} | kl_seq={gen_summary['kl_seq_mean']:.3f} "
          f"kl_tok={gen_summary['kl_tokmean']:.4f} rm={gen_summary['rm_score']['mean']:.3f} "
          f"len={gen_summary['length']['mean']:.1f} | prompts truncated at {max_prompt}: {n_prompt_truncated}/{len(rows)}")
    return {"pairs": pair_summary, "generations": gen_summary, **extra}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the reference policy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--budget", default="none (reference policy)")
    args = ap.parse_args()
    evaluate(args.config, None if args.adapter == "none" else args.adapter, args.name, args.beta, args.mode,
             args.budget)


if __name__ == "__main__":
    main()

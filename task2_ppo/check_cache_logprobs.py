from __future__ import annotations

import argparse
import sys

import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import response_token_logprobs
from common.guards import assert_fixed_files
from common.models import load_policy, load_tokenizer, reference_mode
from common.results import finalize, reconcile, start_run, write_rows
from common.rl_utils import logprobs_eval_mode
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models

TEMPERATURES = (1.0, 0.7)
THRESHOLD = 0.05


def response_ids(tokenizer, row) -> list[int]:
    ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    return ids + ([tokenizer.eos_token_id] if row["terminated_with_eos"] else [])


def prompt_ids(tokenizer, messages, max_prompt_length: int) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"][-max_prompt_length:]


@torch.no_grad()
def compare_row(policy, tokenizer, row, messages, max_prompt_length) -> dict:
    p_ids, r_ids = prompt_ids(tokenizer, messages, max_prompt_length), response_ids(tokenizer, row)
    if len(r_ids) != int(row["response_tokens"]):
        raise ValueError(f"{row['prompt_id']}: rebuilt {len(r_ids)} response tokens, cache has {row['response_tokens']}")
    device = next(policy.parameters()).device
    seq = torch.tensor([p_ids + r_ids], device=device)
    args = (seq, torch.ones_like(seq), len(p_ids), seq[:, len(p_ids):])
    out = {"prompt_id": row["prompt_id"], "n_tokens": len(r_ids)}
    cached = {"old": row["old_logprobs"].float(), "ref": row["ref_logprobs"].float()}
    for t in TEMPERATURES:
        with logprobs_eval_mode(policy):
            old = response_token_logprobs(policy, *args, temperature=t)[0][0].cpu()
            with reference_mode(policy):
                ref = response_token_logprobs(policy, *args, temperature=t)[0][0].cpu()
        out[f"abs_sum_old_T{t}"] = float((old - cached["old"]).abs().sum())
        out[f"abs_sum_ref_T{t}"] = float((ref - cached["ref"]).abs().sum())
    return out


def summarize(rows: list[dict]) -> dict:
    n = sum(r["n_tokens"] for r in rows)
    out = {"n_rollouts": len(rows), "n_tokens": n, "threshold": THRESHOLD}
    for t in TEMPERATURES:
        out[f"mean_abs_old_T{t}"] = sum(r[f"abs_sum_old_T{t}"] for r in rows) / n
        out[f"mean_abs_ref_T{t}"] = sum(r[f"abs_sum_ref_T{t}"] for r in rows) / n
    adopted = min(TEMPERATURES, key=lambda t: out[f"mean_abs_old_T{t}"])
    out["adopted_temperature"] = adopted
    out["pass"] = out[f"mean_abs_old_T{adopted}"] < THRESHOLD and out[f"mean_abs_ref_T{adopted}"] < THRESHOLD
    return out


def main():
    ap = argparse.ArgumentParser(description="Match cached PPO rollout log-probs at T=1.0 vs T=0.7")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    assert_fixed_files(args.mode)
    smoke = args.mode == "smoke"
    run = start_run("results/_smoke/task2_ppo/logprob_gate" if smoke else "results/task2_ppo/logprob_gate",
                    mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))

    cache = torch.load(cfg["cached_rollouts"], map_location="cpu", weights_only=False)[: 2 if smoke else None]
    messages = {r["prompt_id"]: prompt_messages(r) for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, **adapter_kwargs(cfg, cfg["paths"]["ppo_midpoint_policy"]))
    rows = [compare_row(policy, tok, r, messages[r["prompt_id"]], int(cfg["max_prompt_length"])) for r in cache]

    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize)
    reconcile(run["outdir"], {"rows.jsonl", "summary.json"})

    for t in TEMPERATURES:
        print(f"T={t}: mean|old - cached|={summary[f'mean_abs_old_T{t}']:.4f}  "
              f"mean|ref - cached|={summary[f'mean_abs_ref_T{t}']:.4f}")
    if smoke:
        print("smoke: tiny random model, the match is not evaluated")
        return
    t = summary["adopted_temperature"]
    if not summary["pass"]:
        print(f"BREACH: best match T={t} is not below {THRESHOLD}; check prompt rendering and truncation first")
        sys.exit(1)
    print(f"PASS: adopt logprob_temperature={t}")


if __name__ == "__main__":
    main()

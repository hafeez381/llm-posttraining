from __future__ import annotations

import argparse
import sys

import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import response_token_logprobs
from common.guards import assert_fixed_files
from common.models import load_policy, load_tokenizer, reference_mode
from common.results import finalize, reconcile, start_run, write_rows
from common.rl_utils import disable_dropout, logprobs_eval_mode
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task2_ppo.continue_train import RATIO_TOLERANCE, load_critic, response_values

TEMPERATURES = (1.0, 0.7)
ALIGNMENTS = ("prev", "same")
THRESHOLD = 0.05


def response_ids(tokenizer, row) -> list[int]:
    ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    return ids + ([tokenizer.eos_token_id] if row["terminated_with_eos"] else [])


def prompt_ids(tokenizer, messages, max_prompt_length: int, truncation: str = "left") -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[-max_prompt_length:] if truncation == "left" else ids[:max_prompt_length]


@torch.no_grad()
def compare_row(policy, value_model, tokenizer, row, messages, max_prompt_length, truncation) -> dict:
    p_ids, r_ids = prompt_ids(tokenizer, messages, max_prompt_length, truncation), response_ids(tokenizer, row)
    if len(r_ids) != int(row["response_tokens"]):
        raise ValueError(f"{row['prompt_id']}: rebuilt {len(r_ids)} response tokens, cache has {row['response_tokens']}")
    device = next(policy.parameters()).device
    seq = torch.tensor([p_ids + r_ids], device=device)
    args = (seq, torch.ones_like(seq), len(p_ids), seq[:, len(p_ids):])
    out = {"prompt_id": row["prompt_id"], "n_tokens": len(r_ids), "prompt_tokens_untruncated": len(
        tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                  add_special_tokens=False)["input_ids"])}
    cached = {"old": row["old_logprobs"].float(), "ref": row["ref_logprobs"].float(), "values": row["values"].float()}
    for t in TEMPERATURES:
        with logprobs_eval_mode(policy):
            old = response_token_logprobs(policy, *args, temperature=t)[0][0].cpu()
            with reference_mode(policy):
                ref = response_token_logprobs(policy, *args, temperature=t)[0][0].cpu()
        out[f"abs_sum_old_T{t}"] = float((old - cached["old"]).abs().sum())
        out[f"abs_sum_ref_T{t}"] = float((ref - cached["ref"]).abs().sum())
        if t == 1.0:
            train = response_token_logprobs(policy, *args, temperature=t)[0][0].cpu()
            out["ratio_sum_train_vs_eval"] = float(torch.exp(train - old).sum())
            out["n_outside_band"] = int(((train - old).exp() - 1).abs().gt(0.2).sum())
    with logprobs_eval_mode(value_model):
        for a in ALIGNMENTS:
            v = response_values(value_model, *args[:3], len(r_ids), a)[0].cpu()
            out[f"value_max_abs_{a}"] = float((v - cached["values"]).abs().max())
            out[f"value_abs_sum_{a}"] = float((v - cached["values"]).abs().sum())
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
    for a in ALIGNMENTS:
        out[f"value_max_abs_{a}"] = max(r[f"value_max_abs_{a}"] for r in rows)
        out[f"value_mean_abs_{a}"] = sum(r[f"value_abs_sum_{a}"] for r in rows) / n
    out["value_alignment_matched"] = [a for a in ALIGNMENTS if out[f"value_max_abs_{a}"] < THRESHOLD]
    out["ratio_mean_train_vs_eval"] = sum(r["ratio_sum_train_vs_eval"] for r in rows) / n
    out["clip_frac_train_vs_eval"] = sum(r["n_outside_band"] for r in rows) / n
    return out


def main():
    ap = argparse.ArgumentParser(description="Gates on the cached PPO rollouts: log-prob temperature, value "
                                             "alignment, and ratio = 1 before any update")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--dtype", help="override the config dtype; the cache was written in bfloat16")
    ap.add_argument("--truncation", choices=["left", "right"], default="left",
                    help="prompt truncation side; the cache's prompts were right-truncated")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    if args.dtype:
        cfg["dtype"] = args.dtype
    assert_fixed_files(args.mode)
    smoke = args.mode == "smoke"
    name = f"cache_gates_{cfg['dtype']}_{args.truncation}"
    run = start_run(f"results/_smoke/task2_ppo/{name}" if smoke else f"results/task2_ppo/{name}",
                    mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))

    cache = torch.load(cfg["cached_rollouts"], map_location="cpu", weights_only=False)[: 2 if smoke else None]
    messages = {r["prompt_id"]: prompt_messages(r) for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tok = load_tokenizer(cfg["base_model"])
    policy = disable_dropout(load_policy(cfg, trainable=True, **adapter_kwargs(cfg, cfg["paths"]["ppo_midpoint_policy"])))
    value_model = load_critic(cfg)
    rows = [compare_row(policy, value_model, tok, r, messages[r["prompt_id"]], int(cfg["max_prompt_length"]),
                        args.truncation) for r in cache]

    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"dtype": cfg["dtype"], "truncation": args.truncation})
    reconcile(run["outdir"], {"rows.jsonl", "summary.json"})

    for t in TEMPERATURES:
        print(f"T={t}: mean|old - cached|={summary[f'mean_abs_old_T{t}']:.4f}  "
              f"mean|ref - cached|={summary[f'mean_abs_ref_T{t}']:.4f}")
    for a in ALIGNMENTS:
        print(f"values ({a}): max|v - cached|={summary[f'value_max_abs_{a}']:.4f}  "
              f"mean={summary[f'value_mean_abs_{a}']:.4f}")
    print(f"train-mode vs eval-mode ratio mean={summary['ratio_mean_train_vs_eval']:.6f} "
          f"outside [0.8, 1.2]={summary['clip_frac_train_vs_eval']:.4f}")
    if smoke:
        print("smoke: tiny random models, the matches are not evaluated")
        return
    verdicts = {
        "log-prob temperature": (summary["pass"], f"best match T={summary['adopted_temperature']} is not below {THRESHOLD}"),
        "value alignment": (cfg["value_alignment"] in summary["value_alignment_matched"],
                            f"configured value_alignment={cfg['value_alignment']} is not within {THRESHOLD} "
                            f"(matched: {summary['value_alignment_matched']})"),
        "ratio at the start": (abs(summary["ratio_mean_train_vs_eval"] - 1) <= RATIO_TOLERANCE
                               and summary["clip_frac_train_vs_eval"] == 0, "training forward differs from eval mode"),
    }
    for gate, (ok, why) in verdicts.items():
        print(f"{'PASS' if ok else 'BREACH'}: {gate}" + ("" if ok else f": {why}"))
    if not all(ok for ok, _ in verdicts.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

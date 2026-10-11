from __future__ import annotations

import argparse
import sys

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import batch_generate
from common.guards import assert_configs_frozen, assert_fixed_files
from common.models import clear_gpu, load_policy, load_tokenizer
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

BATCH = 8
PROMPT_CAP = 512


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, **adapter_kwargs(cfg, specs[name]))


def generate_policy(cfg, tok, name: str, rows: list[dict], max_new_tokens: int) -> list[dict]:
    model = load_frozen_policy(cfg, name)
    out = []
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        gen = batch_generate(model, tok, [prompt_messages(r) for r in batch], PROMPT_CAP, max_new_tokens,
                             do_sample=False)
        for r, resp, n, trunc in zip(batch, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(resp)
            out.append({"policy": name, "id": str(r.get("prompt_id", r["source_index"])), "response": resp,
                        "n_tokens": int(n), "truncated": bool(trunc), "pred_final": pred,
                        "gold_final": str(r["gold_final"]), "correct": exact_reward(resp, str(r["gold_final"])) == 1.0,
                        "format_ok": pred is not None, "hash_present": "####" in resp})
    del model
    clear_gpu()
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for p in dict.fromkeys(r["policy"] for r in rows):
        sel = [r for r in rows if r["policy"] == p]
        n_tok = np.array([r["n_tokens"] for r in sel], dtype=float)
        out[p] = {"n": len(sel), "accuracy": float(np.mean([r["correct"] for r in sel])),
                  "format_compliance": float(np.mean([r["format_ok"] for r in sel])),
                  "truncation_rate": float(np.mean([r["truncated"] for r in sel])),
                  "hash_but_unparsed": sum(r["hash_present"] and not r["format_ok"] for r in sel),
                  "length_mean": float(n_tok.mean()), "length_sd": float(n_tok.std(ddof=1)) if len(sel) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser(description="Greedy SFT / RLVR / RLAIF responses on GSM8K-eval or SVAMP, exact-verified")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    frozen_sha = None if smoke else assert_configs_frozen()
    rows = read_jsonl(dataset_path(cfg, args.dataset))[: 4 if smoke else None]
    tok = load_tokenizer(cfg["base_model"])
    max_new = 16 if smoke else int(cfg["math_max_new_tokens"])
    run = start_run(f"results/{'_smoke/' if smoke else ''}task5_feedback/math_{args.dataset}", mode=args.mode, cfg=cfg,
                    seed=int(cfg["seed"]))
    gens = []
    for name in policy_specs(cfg):
        gens += generate_policy(cfg, tok, name, rows, max_new)
        print(f"{name}: {summarize(gens)[name]}", flush=True)
    rows_path = run["outdir"] / "generations.jsonl"
    write_rows(rows_path, gens)
    summary = summarize(gens)
    finalize(run, rows_path, summary, summarize, extra={"dataset": args.dataset, "frozen_at": frozen_sha,
                                                         "decoding": f"greedy, prompt cap {PROMPT_CAP}, {max_new} new tokens"})
    reconcile(run["outdir"], {"generations.jsonl", "summary.json"})
    ids = {p: [r["id"] for r in gens if r["policy"] == p] for p in summary}
    ok = all(v == ids["sft"] for v in ids.values()) and all(s["n"] == len(rows) for s in summary.values())
    print(f"{'PASS' if ok else 'BREACH'}: every policy answers every problem in the same order")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

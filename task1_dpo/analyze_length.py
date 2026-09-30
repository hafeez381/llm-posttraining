from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, preference_responses, prompt_messages_from_preference, read_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.guards import assert_fixed_files
from common.logging_utils import set_seed
from common.metrics import parse_word_limit, word_count
from common.models import load_reward_model, load_tokenizer
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg, resolve_models
from task1_dpo.evaluate import load_eval_policy, score_pairs, summarize_pairs, wilson

MIN_CANDIDATES = 5
LENGTH_RATIO = 1.5
LENGTH_RATIO_FALLBACK = 1.3


def dataset_rows(tokenizer, path: str, name: str) -> list[dict]:
    out = []
    for r in read_jsonl(path):
        yc, yr = preference_responses(r)
        nc = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        nr = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        out.append({"dataset": name, "prompt_id": r["prompt_id"], "chosen_tokens": nc, "rejected_tokens": nr,
                    "length_difference": nc - nr,
                    "score_gap": float(r["score_chosen"]) - float(r["score_rejected"])})
    return out


def summarize_dataset(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["dataset"] for r in rows}):
        d = pd.DataFrame([r for r in rows if r["dataset"] == name])
        out[name] = {"n": len(d), "frac_chosen_longer": float((d.length_difference > 0).mean()),
                     "frac_equal_length": float((d.length_difference == 0).mean()),
                     "spearman_score_gap_vs_length_diff": float(d.score_gap.corr(d.length_difference, method="spearman"))}
    return out


def summarize_stratified(rows: list[dict]) -> dict:
    return {m: summarize_pairs([r for r in rows if r["model"] == m]) for m in sorted({r["model"] for r in rows})}


@torch.no_grad()
def word_limit_rows(model_name, policy, tokenizer, rm, rm_tok, prompts, cfg, n_samples, max_new_tokens) -> list[dict]:
    rows = []
    for s in range(n_samples):
        set_seed(int(cfg["seed"]) + s)
        out = batch_generate(policy, tokenizer, [p["messages"] for p in prompts], int(cfg["eval_max_prompt_length"]),
                             max_new_tokens, **cfg["generation"])
        rm_scores = score_reward_pairs(rm, rm_tok, [p["messages"] for p in prompts], out["responses"],
                                       max_length=int(cfg["reward_max_length"]))
        for i, p in enumerate(prompts):
            limit = parse_word_limit(p["messages"][-1]["content"])
            words = word_count(out["responses"][i])
            rows.append({"model": model_name, "prompt_id": p["prompt_id"], "sample_idx": s, "limit": limit,
                         "words": words, "compliant": words <= limit, "n_tokens": out["response_lengths"][i],
                         "eos": out["terminated_with_eos"][i], "rm_score": float(rm_scores[i]),
                         "response": out["responses"][i]})
    return rows


def summarize_word_limit(rows: list[dict]) -> dict:
    out = {}
    for m in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == m]
        k = sum(r["compliant"] for r in mr)
        out[m] = {"n": len(mr), "compliance": k / len(mr), "compliance_wilson95": wilson(k, len(mr)),
                  "mean_words": float(np.mean([r["words"] for r in mr])),
                  "per_prompt": {p: float(np.mean([r["compliant"] for r in mr if r["prompt_id"] == p]))
                                 for p in sorted({r["prompt_id"] for r in mr})}}
    return out


def candidate_rows(tokenizer, policies, results_prefix, word_rows, eval_rows) -> tuple[list[dict], float]:
    sft = {g["prompt_id"]: g for g in read_jsonl(f"results/{results_prefix}task1_dpo/sft/eval/generations.jsonl")}
    chosen_len = {r["prompt_id"]: len(tokenizer(preference_responses(r)[0], add_special_tokens=False)["input_ids"])
                  - len(tokenizer(preference_responses(r)[1], add_special_tokens=False)["input_ids"]) for r in eval_rows}
    prompt_text = {r["prompt_id"]: prompt_messages_from_preference(r)[-1]["content"] for r in eval_rows}

    def build(ratio):
        out = []
        for pol in policies:
            gens = {g["prompt_id"]: g for g in read_jsonl(f"results/{results_prefix}task1_dpo/{pol}/eval/generations.jsonl")}
            pairs = {p["prompt_id"]: p for p in read_jsonl(f"results/{results_prefix}task1_dpo/{pol}/eval/pairs.jsonl")}
            median_gap = float(np.median([p["score_gap"] for p in pairs.values()]))
            for pid, g in gens.items():
                base = sft[pid]
                if g["rm_score"] <= base["rm_score"]:
                    continue
                triggers = []
                if g["n_tokens"] > ratio * max(base["n_tokens"], 1):
                    triggers.append("length_ratio")
                p = pairs[pid]
                if p["margin"] > 0 and chosen_len[pid] > 0 and p["score_gap"] < median_gap:
                    triggers.append("margin_on_longer_weak_pair")
                if triggers:
                    out.append({"policy": pol, "source": "eval", "prompt_id": pid, "triggers": triggers,
                                "prompt": prompt_text[pid][:600], "sft_response": base["response"][:800],
                                "policy_response": g["response"][:800], "sft_rm": base["rm_score"],
                                "policy_rm": g["rm_score"], "sft_tokens": base["n_tokens"], "policy_tokens": g["n_tokens"]})
            sft_wl = {(r["prompt_id"], r["sample_idx"]): r for r in word_rows if r["model"] == "sft"}
            for r in word_rows:
                if r["model"] != pol:
                    continue
                b = sft_wl[(r["prompt_id"], r["sample_idx"])]
                if r["rm_score"] > b["rm_score"] and not r["compliant"]:
                    out.append({"policy": pol, "source": "word_limit", "prompt_id": r["prompt_id"],
                                "sample_idx": r["sample_idx"], "triggers": ["word_limit_violated"], "limit": r["limit"],
                                "sft_response": b["response"][:800], "policy_response": r["response"][:800],
                                "sft_rm": b["rm_score"], "policy_rm": r["rm_score"], "sft_words": b["words"],
                                "policy_words": r["words"]})
        return out

    rows = build(LENGTH_RATIO)
    reward_quality = sum(any(t != "word_limit_violated" for t in r["triggers"]) for r in rows)
    if reward_quality < MIN_CANDIDATES:
        return build(LENGTH_RATIO_FALLBACK), LENGTH_RATIO_FALLBACK
    return rows, LENGTH_RATIO


def summarize_candidates(rows: list[dict]) -> dict:
    triggers = [t for r in rows for t in r["triggers"]]
    return {"n": len(rows), "by_trigger": {t: triggers.count(t) for t in sorted(set(triggers))},
            "reward_vs_quality": sum(any(t != "word_limit_violated" for t in r["triggers"]) for r in rows),
            "instruction_compliance": sum("word_limit_violated" in r["triggers"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    prefix = "_smoke/" if smoke else ""
    run = start_run(f"results/{prefix}task1_dpo/length_analysis", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))
    out = run["outdir"]
    tokenizer = load_tokenizer(cfg["base_model"])

    data = dataset_rows(tokenizer, cfg["paths"]["dpo_standard_train"], "standard") + \
        dataset_rows(tokenizer, cfg["paths"]["dpo_length_train"], "length_balanced")
    write_rows(out / "dataset_pairs.jsonl", data)

    models = {"sft": None, "standard": f"outputs/{prefix}task1_dpo/standard",
              "length_balanced": f"outputs/{prefix}task1_dpo/length_balanced"}
    stratified_rows_in = read_jsonl(cfg["paths"]["dpo_length_eval"])
    if smoke:
        stratified_rows_in = [next(r for r in stratified_rows_in if r["length_stratum"] == s)
                              for s in ("preferred_longer", "length_matched", "rejected_longer")]
    word_prompts = [{"prompt_id": r["prompt_id"], "messages": r["messages"]}
                    for r in read_jsonl(cfg["paths"]["word_limit_prompts"])]
    if any(parse_word_limit(p["messages"][-1]["content"]) is None for p in word_prompts):
        print("BREACH: a word-limit prompt has no parseable limit")
        sys.exit(1)
    n_samples = 1 if smoke else int(cfg["word_limit_samples"])
    rm, rm_tok = load_reward_model(cfg)

    stratified, word_rows = [], []
    for name, adapter in models.items():
        policy = load_eval_policy(cfg, adapter)
        for r in score_pairs(policy, tokenizer, stratified_rows_in, float(cfg["beta"]),
                             int(cfg["max_sequence_length"]), 2 if smoke else int(cfg["eval_batch_size"])):
            stratified.append({"model": name, **r})
        word_rows += word_limit_rows(name, policy, tokenizer, rm, rm_tok, word_prompts, cfg, n_samples,
                                     32 if smoke else int(cfg["max_generation_tokens"]))
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_rows(out / "stratified_pairs.jsonl", stratified)
    write_rows(out / "word_limit.jsonl", word_rows)

    eval_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    candidates, ratio = candidate_rows(tokenizer, ["standard", "length_balanced"], prefix, word_rows, eval_rows)
    write_rows(out / "qualitative_candidates.jsonl", candidates)

    summaries = {}
    for rows_file, fn, name in [("dataset_pairs.jsonl", summarize_dataset, "dataset_summary.json"),
                                ("stratified_pairs.jsonl", summarize_stratified, "stratified_summary.json"),
                                ("word_limit.jsonl", summarize_word_limit, "word_limit_summary.json"),
                                ("qualitative_candidates.jsonl", summarize_candidates, "candidates_summary.json")]:
        rows = read_jsonl(out / rows_file)
        summaries[name] = fn(rows)
        finalize(run, out / rows_file, summaries[name], fn, name=name,
                 extra={"length_ratio_threshold": ratio} if "candidates" in name else None)
    reconcile(out, {"dataset_pairs.jsonl", "stratified_pairs.jsonl", "word_limit.jsonl",
                    "qualitative_candidates.jsonl", "dataset_summary.json", "stratified_summary.json",
                    "word_limit_summary.json", "candidates_summary.json"})

    for name, d in summaries["dataset_summary.json"].items():
        print(f"data {name}: chosen longer {d['frac_chosen_longer']:.3f}, spearman(score gap, length diff) "
              f"{d['spearman_score_gap_vs_length_diff']:.3f}")
    for m, s in summaries["stratified_summary.json"].items():
        print(f"{m}: " + ", ".join(f"{k} acc={v['accuracy']:.3f} (n={v['n']})" for k, v in s["by_stratum"].items()))
    for m, s in summaries["word_limit_summary.json"].items():
        print(f"{m}: word-limit compliance {s['compliance']:.3f} over {s['n']} samples, mean words {s['mean_words']:.1f}")
    c = summaries["candidates_summary.json"]
    print(f"candidates: {c['by_trigger']} (length ratio threshold {ratio})")

    counts = {m: s["n"] for m, s in summaries["word_limit_summary.json"].items()}
    breaches = [f"{m}: {n} word-limit rows, expected {10 * n_samples}" for m, n in counts.items() if n != 10 * n_samples]
    if not smoke:
        breaches += [f"{m} {k}: n={v['n']} != 82" for m, s in summaries["stratified_summary.json"].items()
                     for k, v in s["by_stratum"].items() if v["n"] != 82]
        breaches += [f"only {c[k]} {k} candidates" for k in ("reward_vs_quality", "instruction_compliance")
                     if c[k] < MIN_CANDIDATES]
    if breaches:
        print("BREACH:\n  " + "\n  ".join(breaches))
        sys.exit(1)
    print("PASS: stratum counts, word-limit sample counts and candidate counts")


if __name__ == "__main__":
    main()

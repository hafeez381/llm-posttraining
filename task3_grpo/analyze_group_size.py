from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl
from common.guards import assert_fixed_files
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg
from task3_grpo.grpo import group_relative_advantages, group_stats

BINS = ("hard", "medium", "easy")
TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    if 8 % k:
        raise ValueError(f"K={k} does not divide the 8 cached completions")
    groups = []
    for source_index, rows in by_prompt.items():
        rows = rows[:8]
        for start in range(0, 8, k):
            block = rows[start:start + k]
            groups.append({
                "source_index": source_index,
                "prompt_id": block[0]["prompt_id"],
                "gen_indices": [int(r["generation_index"]) for r in block],
                "rewards": [float(r["reward"]) for r in block],
            })
    return groups


def exhaustive_subset_stats(rewards_8, k: int, tol: float = TOL) -> dict:
    r = np.asarray(rewards_8, dtype=float)
    stds = np.array([r[list(c)].std() for c in itertools.combinations(range(len(r)), k)])
    return {"n_subsets": len(stds), "informative_frac": float((stds >= tol).mean()), "mean_std": float(stds.mean())}


def difficulty_bins(by_prompt) -> tuple[dict, list]:
    means = {p: float(np.mean([r["reward"] for r in rows[:8]])) for p, rows in by_prompt.items()}
    order = sorted(means, key=lambda p: (means[p], int(p) if p.isdigit() else p))
    n = len(order)
    if n % 3:
        raise ValueError(f"{n} prompts cannot be split into equal thirds")
    size = n // 3
    bins = {p: BINS[i // size] for i, p in enumerate(order)}
    ties = [(order[i - 1], order[i]) for i in range(size, n, size) if means[order[i - 1]] == means[order[i]]]
    return bins, ties


def build_rows(by_prompt, ks, bins) -> list[dict]:
    rows = []
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k)
        rewards = torch.tensor([x for g in groups for x in g["rewards"]], dtype=torch.float64)
        gids = torch.arange(len(groups)).repeat_interleave(k)
        adv = group_relative_advantages(rewards, gids, eps=TOL)
        for i, (g, st) in enumerate(zip(groups, group_stats(rewards, gids, tol=TOL))):
            a = adv[gids == i]
            rows.append({
                "kind": "group", "k": k, "source_index": g["source_index"], "prompt_id": g["prompt_id"],
                "bin": bins[g["source_index"]], "gen_indices": g["gen_indices"], "rewards": g["rewards"],
                "mean": st["mean"], "std": st["std"], "informative": st["informative"],
                "informative_isclose": st["informative_isclose"],
                "centered_sq_sum": float(sum(c * c for c in st["centered"])),
                "n_nonzero_adv": int((a.abs() > 0).sum()),
                "adv_var": float(a.var(unbiased=False)),
            })
        for p, prompt_rows in by_prompt.items():
            rows.append({"kind": "prompt_subsets", "k": k, "source_index": p, "bin": bins[p],
                         **exhaustive_subset_stats([r["reward"] for r in prompt_rows[:8]], k)})
    return rows


def _group_metrics(groups: list[dict]) -> dict:
    n_gen = sum(len(g["rewards"]) for g in groups)
    return {
        "n_groups": len(groups),
        "n_generations": n_gen,
        "informative_rate": float(np.mean([g["informative"] for g in groups])),
        "n_uninformative": sum(not g["informative"] for g in groups),
        "n_uninformative_isclose": sum(not g["informative_isclose"] for g in groups),
        "mean_within_group_std": float(np.mean([g["std"] for g in groups])),
        "centered_reward_var": sum(g["centered_sq_sum"] for g in groups) / n_gen,
        "frac_nonzero_adv": sum(g["n_nonzero_adv"] for g in groups) / n_gen,
    }


def _subset_metrics(prompts: list[dict]) -> dict:
    return {"exhaustive_informative_frac": float(np.mean([p["informative_frac"] for p in prompts])),
            "exhaustive_mean_std": float(np.mean([p["mean_std"] for p in prompts]))}


def summarize(rows: list[dict]) -> dict:
    out = {"tolerance": TOL, "bins": {b: sorted({r["source_index"] for r in rows if r["bin"] == b}) for b in BINS}}
    for k in sorted({r["k"] for r in rows}):
        groups = [r for r in rows if r["kind"] == "group" and r["k"] == k]
        prompts = [r for r in rows if r["kind"] == "prompt_subsets" and r["k"] == k]
        per_k = {**_group_metrics(groups), **_subset_metrics(prompts),
                 "tolerance_disagreements": [f'{g["source_index"]}:{g["gen_indices"]}' for g in groups
                                             if g["informative"] != g["informative_isclose"]],
                 "by_bin": {}}
        for b in BINS:
            gb = [g for g in groups if g["bin"] == b]
            pb = [p for p in prompts if p["bin"] == b]
            if gb:
                per_k["by_bin"][b] = {**_group_metrics(gb), **_subset_metrics(pb)}
        out[f"K{k}"] = per_k
    return out


def check_gates(rows, summary, ks, n_prompts, ties) -> list[str]:
    breaches = []
    for k in ks:
        s = summary[f"K{k}"]
        if s["n_generations"] != 8 * n_prompts:
            breaches.append(f"K={k}: {s['n_generations']} generations, expected {8 * n_prompts}")
        if s["n_groups"] != n_prompts * 8 // k:
            breaches.append(f"K={k}: {s['n_groups']} groups, expected {n_prompts * 8 // k}")
    for r in rows:
        if r["kind"] == "group" and r["informative"] and abs(r["adv_var"] - 1.0) > 1e-5:
            breaches.append(f"K={r['k']} {r['source_index']}{r['gen_indices']}: advantage variance {r['adv_var']:.6f} != 1")
    sizes = {b: len(v) for b, v in summary["bins"].items()}
    if len(set(sizes.values())) != 1:
        breaches.append(f"unequal bins {sizes}")
    if ties:
        print(f"note: reward ties at bin boundaries, broken by source_index: {ties}")
    return breaches


def qualitative_rescues(by_prompt, rows) -> list[dict]:
    parent = {r["source_index"]: r["informative"] for r in rows if r["kind"] == "group" and r["k"] == 8}
    out = []
    for r in rows:
        if r["kind"] == "group" and r["k"] == 2 and not r["informative"] and parent[r["source_index"]]:
            comps = {int(c["generation_index"]): c["completion"] for c in by_prompt[r["source_index"]]}
            out.append({"source_index": r["source_index"], "gen_indices": r["gen_indices"], "rewards": r["rewards"],
                        "k8_rewards": [float(c["reward"]) for c in by_prompt[r["source_index"]][:8]],
                        "completions": [comps[i] for i in r["gen_indices"]]})
    return out


def main():
    ap = argparse.ArgumentParser(description="Equal-generation group-size study on the K=8 reward cache")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert_fixed_files(args.mode)

    by_prompt = load_k8_cache(cfg["group_cache"])
    if args.mode == "smoke":
        by_prompt = dict(list(by_prompt.items())[:6])
    ks = [int(k) for k in cfg["group_sizes"]]
    outdir = "results/_smoke/task3_grpo/group_size" if args.mode == "smoke" else "results/task3_grpo/group_size"
    run = start_run(outdir, mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))

    bins, ties = difficulty_bins(by_prompt)
    rows = build_rows(by_prompt, ks, bins)
    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, rows)
    rescues = qualitative_rescues(by_prompt, rows)
    write_rows(run["outdir"] / "rescued_k2_groups.jsonl", rescues)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize)
    reconcile(run["outdir"], {"rows.jsonl", "summary.json", "rescued_k2_groups.jsonl"})

    print(f"prompts={len(by_prompt)} bins={ {b: len(v) for b, v in summary['bins'].items()} }")
    for k in ks:
        s = summary[f"K{k}"]
        print(f"K={k}: groups={s['n_groups']} gens={s['n_generations']} informative={s['informative_rate']:.4f} "
              f"(uninformative {s['n_uninformative']}, isclose {s['n_uninformative_isclose']}) "
              f"within_std={s['mean_within_group_std']:.4f} centered_var={s['centered_reward_var']:.4f} "
              f"nonzero_adv={s['frac_nonzero_adv']:.4f} exhaustive_informative={s['exhaustive_informative_frac']:.4f}")
        for b, v in s["by_bin"].items():
            print(f"    {b:6s} informative={v['informative_rate']:.4f} within_std={v['mean_within_group_std']:.4f} "
                  f"centered_var={v['centered_reward_var']:.4f} exhaustive_informative={v['exhaustive_informative_frac']:.4f}")
    print(f"rescued K=2 groups: {len(rescues)}")

    breaches = check_gates(rows, summary, ks, len(by_prompt), ties)
    if breaches:
        print("BREACH:\n  " + "\n  ".join(breaches))
        sys.exit(1)
    print("PASS: equal generations per K, expected group counts, equal bins, unit advantage variance in informative groups")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.guards import GuardError, assert_matched_budget
from common.logging_utils import load_json
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg
from task2_ppo.evaluate import evaluate
from task3_grpo.continue_train import run_grpo

LOSS_TYPES = ("grpo", "dr_grpo")
NOISE_FLOOR = 1e-2


def run_fork(config: str, cfg: dict, loss_type: str, mode: str) -> dict:
    prefix = "_smoke/" if mode == "smoke" else ""
    name = f"fork_{loss_type}"
    res = repo_path(f"results/{prefix}task3_grpo/{name}")
    if not (res / "train" / "summary.json").exists():
        run_grpo(config, updates=int(cfg["fork_updates"]), loss_type=loss_type, run_name=name, mode=mode)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if not (res / "eval" / "summary.json").exists():
        evaluate(config, f"outputs/{prefix}task3_grpo/{name}", name, task="task3_grpo", mode=mode)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    train, ev = load_json(res / "train" / "summary.json"), load_json(res / "eval" / "summary.json")
    t, e = train["summary"], ev["summary"]
    updates = read_jsonl(res / "train" / "updates.jsonl")
    comp = [(a, n, w) for r in updates
            for a, n, w in zip(r["adv_per_completion"], r["len_per_completion"], r["weight_per_completion"])]
    side = lambda keep: float(np.mean([n for a, n, _ in comp if keep(a)])) if any(keep(a) for a, _, _ in comp) else None
    return {
        "name": name, "loss_type": loss_type,
        "updates": t["n_updates"], "k": train["config"]["num_generations"],
        "cap": train["config"]["max_completion_length"], "seed": train["seed"], "prompt_ids": t["prompt_ids"],
        "generated_tokens": t["generated_tokens"], "train_len_mean": t["len_mean"],
        "train_len_by_update": [r["len_mean"] for r in updates],
        "train_len_adv_pos": side(lambda a: a > 0), "train_len_adv_neg": side(lambda a: a < 0),
        "weight_len_pairs": [[w, n] for _, n, w in comp],
        "train_reward_mean": t["reward_mean"], "train_kl_mean": t["kl_mean"],
        "eval_rm_mean": e["rm_score"]["mean"], "eval_rm_se": e["rm_score"]["boot_se"],
        "eval_kl_seq_mean": e["kl_seq_mean"], "eval_kl_seq_se": e["kl_seq"]["boot_se"],
        "eval_length_mean": e["length"]["mean"], "eval_length_se": e["length"]["boot_se"],
        "eval_length_iqr": e["length"]["iqr"], "eval_entropy_tokmean": e["entropy_exact_tokmean"],
        "eval_truncation_rate": e["truncation_rate"],
        "train_git_sha": train["git_sha"], "eval_git_sha": ev["git_sha"],
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        w, n = np.array(r["weight_len_pairs"]).T if r["weight_len_pairs"] else (np.array([]), np.array([]))
        keep = w > 0
        out[r["name"]] = {
            "generated_tokens": r["generated_tokens"],
            "weight_length_corr": float(np.corrcoef(w[keep], n[keep])[0, 1]) if keep.sum() > 1 else float("nan"),
            "eval_length_mean": r["eval_length_mean"], "eval_rm_mean": r["eval_rm_mean"],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Canonical vs Dr. GRPO normalization forks from the GRPO midpoint")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    prefix = "_smoke/" if args.mode == "smoke" else ""
    rows = [run_fork(args.config, cfg, lt, args.mode) for lt in LOSS_TYPES]

    run = start_run(f"results/{prefix}task3_grpo/normalization_study", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))
    rows_path = run["outdir"] / "forks.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)

    probe = None
    std_log = repo_path(f"results/{prefix}task3_grpo/standard/train/updates.jsonl")
    if std_log.exists():
        fork = read_jsonl(f"results/{prefix}task3_grpo/fork_grpo/train/updates.jsonl")
        std = read_jsonl(std_log)[: len(fork)]
        diffs = [abs(a["reward_mean"] - b["reward_mean"]) for a, b in zip(fork, std)]
        probe = {"reward_mean_abs_diff": diffs, "noise_floor": max(diffs) > NOISE_FLOOR}
    finalize(run, rows_path, summary, summarize, extra={"canonical_fork_vs_standard": probe,
                                                         "budget": f"{cfg['fork_updates']} updates per fork"})
    reconcile(run["outdir"], {"forks.jsonl", "summary.json"})

    for r in rows:
        print(f"{r['name']}: rm={r['eval_rm_mean']:.3f}±{r['eval_rm_se']:.3f} kl={r['eval_kl_seq_mean']:.3f} "
              f"len={r['eval_length_mean']:.1f}±{r['eval_length_se']:.1f} train len={r['train_len_mean']:.1f} "
              f"(A>0 {r['train_len_adv_pos']}, A<0 {r['train_len_adv_neg']}) tokens={r['generated_tokens']} "
              f"weight-length corr={summary[r['name']]['weight_length_corr']:.3f}")
    print(f"canonical fork vs standard: {probe}")
    try:
        assert_matched_budget(rows)
    except GuardError as e:
        print(f"BREACH: {e}")
        sys.exit(1)
    print("PASS: forks share update count, prompt sequence, K, cap and seed")


if __name__ == "__main__":
    main()

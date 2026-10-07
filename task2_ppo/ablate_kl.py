from __future__ import annotations

import argparse
import gc
import sys

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.guards import GuardError, assert_matched_budget
from common.logging_utils import load_json
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate

NOISE_FLOOR = 1e-2


def fork_name(cfg: dict, eps: float, beta: float) -> str:
    if eps != float(cfg["clip_epsilon"]):
        return f"eps_{eps:g}"
    if beta != float(cfg["kl_beta"]):
        return f"kl_{beta:g}"
    return f"eps_{eps:g}_kl_{beta:g}"


def run_fork(config: str, cfg: dict, eps: float, beta: float, mode: str) -> dict:
    prefix = "_smoke/" if mode == "smoke" else ""
    name = fork_name(cfg, eps, beta)
    res = repo_path(f"results/{prefix}task2_ppo/{name}")
    if not (res / "train" / "summary.json").exists():
        run_ppo(config, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name, mode=mode)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if not (res / "eval" / "summary.json").exists():
        evaluate(config, f"outputs/{prefix}task2_ppo/{name}", name, mode=mode)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    train, ev = load_json(res / "train" / "summary.json"), load_json(res / "eval" / "summary.json")
    t, e = train["summary"], ev["summary"]
    return {
        "name": name, "clip_epsilon": eps, "kl_beta": beta,
        "updates": t["n_updates"], "k": train["config"]["prompts_per_update"],
        "cap": train["config"]["max_response_length"], "seed": train["seed"], "prompt_ids": t["prompt_ids"],
        "generated_tokens": t["generated_tokens"], "approx_kl_post_mean": t["approx_kl_post_mean"],
        "approx_kl_post_max": t["approx_kl_post_max"], "grad_norm_policy_max": t["grad_norm_policy_max"],
        "grad_clipped_policy_frac": t["grad_clipped_policy_frac"], "train_reward_eff_mean": t["reward_eff_mean"],
        "eval_rm_mean": e["rm_score"]["mean"], "eval_rm_se": e["rm_score"]["boot_se"],
        "eval_kl_seq_mean": e["kl_seq_mean"], "eval_kl_seq_se": e["kl_seq"]["boot_se"],
        "eval_length_mean": e["length"]["mean"], "eval_length_se": e["length"]["boot_se"],
        "eval_entropy_tokmean": e["entropy_exact_tokmean"], "eval_eos_rate": e["eos_rate"],
        "eval_truncation_rate": e["truncation_rate"],
        "train_git_sha": train["git_sha"], "eval_git_sha": ev["git_sha"],
    }


def summarize(rows: list[dict]) -> dict:
    return {"forks": [r["name"] for r in rows],
            "generated_tokens": {r["name"]: r["generated_tokens"] for r in rows},
            "approx_kl_post_max": {r["name"]: r["approx_kl_post_max"] for r in rows},
            "all_approx_kl_post_below_1e-5": all(r["approx_kl_post_max"] < 1e-5 for r in rows)}


def run_study(study: str, config: str, mode: str) -> None:
    cfg = load_yaml(config)
    prefix = "_smoke/" if mode == "smoke" else ""
    if study == "kl":
        conditions = [(float(cfg["clip_epsilon"]), float(b)) for b in cfg["kl_values"]]
    else:
        conditions = [(float(e), float(cfg["kl_beta"])) for e in cfg["clip_values"]]
    rows = [run_fork(config, cfg, eps, beta, mode) for eps, beta in conditions]

    run = start_run(f"results/{prefix}task2_ppo/{study}_study", mode=mode, cfg=cfg, seed=int(cfg["seed"]))
    rows_path = run["outdir"] / "forks.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)

    shared = fork_name(cfg, float(cfg["clip_epsilon"]), float(cfg["kl_beta"]))
    probe = None
    std_log = repo_path(f"results/{prefix}task2_ppo/standard/train/updates.jsonl")
    if std_log.exists():
        fork = read_jsonl(f"results/{prefix}task2_ppo/{shared}/train/updates.jsonl")[:2]
        std = read_jsonl(std_log)[:2]
        diffs = [abs(a["reward_eff"] - b["reward_eff"]) for a, b in zip(fork, std)]
        probe = {"reward_eff_abs_diff_updates_1_2": diffs, "noise_floor": max(diffs) > NOISE_FLOOR,
                 "same_responses": [a["resp_len"] == b["resp_len"] for a, b in zip(fork, std)]}
    finalize(run, rows_path, summary, summarize, extra={"shared_fork": shared, "shared_fork_probe": probe,
                                                         "budget": f"{cfg['fork_updates']} updates per fork"})
    reconcile(run["outdir"], {"forks.jsonl", "summary.json"})

    for r in rows:
        print(f"{r['name']}: rm={r['eval_rm_mean']:.3f}±{r['eval_rm_se']:.3f} kl={r['eval_kl_seq_mean']:.3f}"
              f"±{r['eval_kl_seq_se']:.3f} len={r['eval_length_mean']:.1f} ent={r['eval_entropy_tokmean']:.3f} "
              f"akl_post max={r['approx_kl_post_max']:.2e} tokens={r['generated_tokens']}"
              + (" (shared)" if r["name"] == shared else ""))
    print(f"shared fork vs standard, updates 1-2: {probe}")
    if summary["all_approx_kl_post_below_1e-5"]:
        print("NOTE: post-update KL < 1e-5 for every fork; the updates are too small for this knob to bind")
    try:
        assert_matched_budget(rows)
    except GuardError as e:
        print(f"BREACH: {e}")
        sys.exit(1)
    print("PASS: forks share update count, prompt sequence, prompts per update, cap and seed")


def main():
    ap = argparse.ArgumentParser(description="KL-pressure forks from the PPO midpoint")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    run_study("kl", args.config, args.mode)


if __name__ == "__main__":
    main()

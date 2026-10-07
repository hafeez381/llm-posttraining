from __future__ import annotations

import argparse
import sys

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg

POLICIES = ("standard", "eps_0.05", "eps_0.2_kl_0.1", "eps_0.5", "kl_0", "kl_0.2")
REPEAT_THRESHOLD = 0.2
LENGTH_RATIO = 1.5
SAME_LENGTH = 0.1
MIN_CANDIDATES = 3


def repeated_4gram_fraction(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def build_rows(base: str) -> list[dict]:
    mid = {r["prompt_id"]: r for r in read_jsonl(f"{base}/midpoint/eval/generations.jsonl")}
    rows = []
    for policy in POLICIES:
        gens = read_jsonl(f"{base}/{policy}/eval/generations.jsonl")
        delta = np.array([g["rm_score"] - mid[g["prompt_id"]]["rm_score"] for g in gens])
        sd = float(delta.std(ddof=1))
        for g, d in zip(gens, delta):
            m = mid[g["prompt_id"]]
            ratio = g["n_tokens"] / max(m["n_tokens"], 1)
            rep = repeated_4gram_fraction(g["response"])
            flags = {"longer": ratio >= LENGTH_RATIO, "newly_truncated": g["truncated"] and not m["truncated"],
                     "repetitive": rep > REPEAT_THRESHOLD}
            up = d > sd
            rows.append({
                "policy": policy, "prompt_id": g["prompt_id"], "rm": g["rm_score"], "rm_midpoint": m["rm_score"],
                "delta_rm": float(d), "delta_rm_sd": sd, "length": g["n_tokens"], "length_midpoint": m["n_tokens"],
                "length_ratio": ratio, "repeat4": rep, "kl_seq": g["kl_seq"], **flags,
                "candidate": ("diverge" if up and any(flags.values())
                              else "together" if up and abs(ratio - 1) <= SAME_LENGTH else None),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for policy in dict.fromkeys(r["policy"] for r in rows):
        sel = [r for r in rows if r["policy"] == policy]
        out[policy] = {
            "delta_rm_mean": float(np.mean([r["delta_rm"] for r in sel])),
            "length_ratio_median": float(np.median([r["length_ratio"] for r in sel])),
            "newly_truncated": sum(r["newly_truncated"] for r in sel),
            "repetitive": sum(r["repetitive"] for r in sel),
            "rm_length_corr": float(np.corrcoef([r["rm"] for r in sel], [r["length"] for r in sel])[0, 1]),
            "n_diverge": sum(r["candidate"] == "diverge" for r in sel),
            "n_together": sum(r["candidate"] == "together" for r in sel),
        }
    out["total"] = {k: sum(v[k] for v in out.values()) for k in ("n_diverge", "n_together")}
    return out


def main():
    ap = argparse.ArgumentParser(description="Held-out cases where learned reward and response quality agree or "
                                             "disagree, against the midpoint")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = f"results/{'_smoke/' if args.mode == 'smoke' else ''}task2_ppo"
    run = start_run(f"{base}/overoptimization", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))
    rows = build_rows(str(repo_path(base)))
    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"rule": {
        "diverge": f"delta RM > 1 SD and (length x{LENGTH_RATIO}, newly truncated, or repeated-4-gram fraction > "
                   f"{REPEAT_THRESHOLD})", "together": f"delta RM > 1 SD and length within {SAME_LENGTH:.0%}"}})
    reconcile(run["outdir"], {"rows.jsonl", "summary.json"})
    for policy, s in summary.items():
        print(policy, s)
    for kind in ("diverge", "together"):
        ok = summary["total"][f"n_{kind}"] >= MIN_CANDIDATES
        print(f"{'PASS' if ok else 'BREACH'}: at least {MIN_CANDIDATES} '{kind}' candidates")
    if args.mode == "full" and min(summary["total"].values()) < MIN_CANDIDATES:
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from collections import Counter

from common.data import load_yaml, read_jsonl
from common.logging_utils import load_json
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg

CONFOUNDS = ["RLVR was trained for 80 steps and RLAIF for 50 (supplied checkpoints), so the two differ in budget as "
             "well as feedback source."]


def failure_type(r: dict) -> str:
    if r["correct"]:
        return "correct"
    if r["truncated"]:
        return "truncated"
    if not r["hash_present"]:
        return "no_designated_final"
    if not r["format_ok"]:
        return "hash_unparsed"
    return "wrong_final"


def summarize(rows: list[dict]) -> dict:
    return {r["policy"]: {k: v for k, v in r.items() if k != "policy"} for r in rows}


def main():
    ap = argparse.ArgumentParser(description="RLVR vs RLAIF: in-domain, out-of-domain and pairwise results side by side")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = f"results/{'_smoke/' if args.mode == 'smoke' else ''}task5_feedback"
    gsm, svamp = (load_json(f"{base}/math_{d}/summary.json")["summary"] for d in ("gsm", "transfer"))
    pairwise = load_json(f"{base}/pairwise/summary.json")["summary"]
    transfer_rows = read_jsonl(f"{base}/math_transfer/generations.jsonl")
    run = start_run(f"{base}/comparison", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))

    rows = []
    for p in gsm:
        failures = Counter(failure_type(r) for r in transfer_rows if r["policy"] == p)
        row = {"policy": p, "gsm_accuracy": gsm[p]["accuracy"], "svamp_accuracy": svamp[p]["accuracy"],
               "drop_abs": gsm[p]["accuracy"] - svamp[p]["accuracy"],
               "drop_rel": (gsm[p]["accuracy"] - svamp[p]["accuracy"]) / gsm[p]["accuracy"] if gsm[p]["accuracy"] else None,
               "gsm_format": gsm[p]["format_compliance"], "svamp_format": svamp[p]["format_compliance"],
               "gsm_length_mean": gsm[p]["length_mean"], "svamp_length_mean": svamp[p]["length_mean"],
               "svamp_failure_types": dict(failures), "svamp_n": svamp[p]["n"]}
        for ds in ("gsm", "transfer"):
            if f"{ds}|{p}" in pairwise:
                row[f"{ds}_win_rate_vs_sft"] = pairwise[f"{ds}|{p}"]["win_rate"]
        rows.append(row)
    rows_path = run["outdir"] / "task5_summary_rows.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"confounds": CONFOUNDS})
    reconcile(run["outdir"], {"task5_summary_rows.jsonl", "summary.json"})
    for r in rows:
        print(r)
    ok = all(sum(r["svamp_failure_types"].values()) == r["svamp_n"] for r in rows)
    print(f"{'PASS' if ok else 'BREACH'}: SVAMP failure types cover every response per policy")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

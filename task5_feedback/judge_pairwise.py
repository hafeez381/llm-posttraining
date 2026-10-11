from __future__ import annotations

import argparse
import sys

from common.data import load_yaml, read_jsonl
from common.guards import assert_fixed_files
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg, resolve_models
from task5_feedback.evaluate_math import dataset_path
from task5_feedback.judge_recording import RecordingPairwiseJudge

SCORE = {"A": 1.0, "TIE": 0.5, "B": 0.0}
JUDGE_PREF = {"A": "policy", "TIE": "tie", "B": "sft"}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ds, pol in dict.fromkeys((r["dataset"], r["policy"]) for r in rows):
        sel = [r for r in rows if r["dataset"] == ds and r["policy"] == pol]
        conf = {v: {j: sum(r["verifier_pref"] == v and JUDGE_PREF[r["pref"]] == j for r in sel)
                    for j in ("policy", "tie", "sft")} for v in ("policy", "tie", "sft")}
        decisive = [r for r in sel if r["verifier_pref"] != "tie"]
        vties = [r for r in sel if r["verifier_pref"] == "tie"]
        out[f"{ds}|{pol}"] = {
            "n": len(sel), "win_rate": sum(SCORE[r["pref"]] for r in sel) / len(sel),
            "n_win": sum(r["pref"] == "A" for r in sel), "n_tie": sum(r["pref"] == "TIE" for r in sel),
            "n_loss": sum(r["pref"] == "B" for r in sel), "n_unparsed": sum(not r["parsed"] for r in sel),
            "n_identical": sum(r["identical_skip"] for r in sel),
            "confusion_verifier_rows_judge_cols": conf,
            "agreement_when_verifier_decisive": (sum(JUDGE_PREF[r["pref"]] == r["verifier_pref"] for r in decisive)
                                                 / len(decisive)) if decisive else float("nan"),
            "judge_decisive_rate_on_verifier_ties": (sum(r["pref"] != "TIE" for r in vties) / len(vties))
            if vties else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="RLVR and RLAIF vs SFT with the fixed pairwise AI judge")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    base = f"results/{'_smoke/' if smoke else ''}task5_feedback"
    run = start_run(f"{base}/pairwise", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))
    judge = RecordingPairwiseJudge(cfg, f"{base}/judge_cache/pairwise_cache.json",
                                   f"{base}/judge_cache/pairwise_raw.json")
    rows = []
    for ds in ("gsm", "transfer"):
        question = {str(r.get("prompt_id", r["source_index"])): r["question"] for r in read_jsonl(dataset_path(cfg, ds))}
        gens = read_jsonl(f"{base}/math_{ds}/generations.jsonl")
        sft = {g["id"]: g for g in gens if g["policy"] == "sft"}
        for g in gens:
            if g["policy"] == "sft":
                continue
            s = sft[g["id"]]
            identical = g["response"] == s["response"]
            rec = ({"pref": "TIE", "raw": "", "parsed": True, "swapped": False} if identical
                   else judge.compare_recorded(question[g["id"]], g["response"], s["response"]))
            v = int(g["correct"]) - int(s["correct"])
            rows.append({"dataset": ds, "policy": g["policy"], "id": g["id"], **rec, "identical_skip": identical,
                         "verifier_pref": "policy" if v > 0 else "sft" if v < 0 else "tie"})
        print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} pairs judged", flush=True)
    rows_path = run["outdir"] / "pairs.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"judge_model": cfg["ai_judge_model"]})
    reconcile(run["outdir"], {"pairs.jsonl", "summary.json"})
    for k, s in summary.items():
        print(f"{k}: win_rate={s['win_rate']:.3f} W/T/L={s['n_win']}/{s['n_tie']}/{s['n_loss']} "
              f"unparsed={s['n_unparsed']} identical={s['n_identical']} agree(verifier decisive)="
              f"{s['agreement_when_verifier_decisive']:.3f} judge decisive on verifier ties="
              f"{s['judge_decisive_rate_on_verifier_ties']:.3f}")
    ok = all(sum(sum(r.values()) for r in s["confusion_verifier_rows_judge_cols"].values()) == s["n"]
             for s in summary.values())
    print(f"{'PASS' if ok else 'BREACH'}: each verifier-judge table covers every pair")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

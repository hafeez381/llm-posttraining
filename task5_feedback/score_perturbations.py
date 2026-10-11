from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from common.guards import assert_fixed_files
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg, resolve_models
from task5_feedback.judge_recording import RecordingPairwiseJudge
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

ANCHOR = "clean_correct"
CATEGORIES = {
    "corrupt_reasoning_correct_final": "reasoning",
    "good_reasoning_wrong_final": "outcome",
    "gold_distractor_wrong_final": "outcome_distractor",
    "persuasive_filler_correct": "filler",
    "clean_correct": "identity_control",
}

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome(pref_for_clean: str, category: str) -> str:
    if pref_for_clean == "tie":
        return "tie"
    return "better" if pref_for_clean == "clean" and category != "identity_control" else "wrong"


def summarize(rows: list[dict]) -> dict:
    out = {}
    for mech in ("verifier", "judge"):
        out[mech] = {}
        for cat in dict.fromkeys(CATEGORIES.values()):
            sel = [r for r in rows if r["mechanism"] == mech and r["category"] == cat]
            out[mech][cat] = {k: sum(r["outcome"] == k for r in sel) / len(sel) for k in ("better", "tie", "wrong")}
            out[mech][cat]["n"] = len(sel)
        out[mech]["S_reason"] = out[mech]["reasoning"]["better"]
        out[mech]["S_outcome"] = out[mech]["outcome"]["better"]
        pooled = [r for r in rows if r["mechanism"] == mech and r["category"] in ("outcome", "outcome_distractor")]
        out[mech]["S_outcome_pooled"] = sum(r["outcome"] == "better" for r in pooled) / len(pooled)
    judge = defaultdict(dict)
    for r in rows:
        if r["mechanism"] == "judge":
            judge[(r["problem_id"], r["category"])][r["order"]] = r["pref_for_clean"]
    out["judge"]["position_consistency"] = sum(v["clean_first"] == v["clean_second"] for v in judge.values()) / len(judge)
    out["judge"]["n_unparsed"] = sum(not r["parsed"] for r in rows if r["mechanism"] == "judge")
    return out


def main():
    ap = argparse.ArgumentParser(description="Verifier vs AI judge on the controlled diagnostic pairs")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    base = f"results/{'_smoke/' if smoke else ''}task5_feedback"
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    if smoke:
        groups = dict(list(groups.items())[:2])
    run = start_run(f"{base}/diagnostics", mode=args.mode, cfg=cfg, seed=int(cfg["seed"]))
    judge = RecordingPairwiseJudge(cfg, f"{base}/judge_cache/diagnostic_cache.json",
                                   f"{base}/judge_cache/diagnostic_raw.json")
    rows, verifier_ok = [], True
    for pid, v in groups.items():
        clean = v[ANCHOR]
        q, gold = clean["question"], str(clean["gold_final"])
        for variant, cat in CATEGORIES.items():
            other = v[variant]
            rc, ro = exact_reward(clean["response"], gold), exact_reward(other["response"], gold)
            verifier_ok &= rc == float(clean["expected_exact_reward"]) and ro == float(other["expected_exact_reward"])
            pref = "tie" if rc == ro else "clean" if rc > ro else "other"
            rows.append({"problem_id": pid, "category": cat, "variant": variant, "mechanism": "verifier",
                         "order": "none", "pref_for_clean": pref, "parsed": True, "raw": "",
                         "outcome": outcome(pref, cat)})
            for order, (a, b) in (("clean_first", (clean, other)), ("clean_second", (other, clean))):
                rec = judge.compare_recorded(q, a["response"], b["response"])
                clean_side = "A" if order == "clean_first" else "B"
                pref = "tie" if rec["pref"] == "TIE" else "clean" if rec["pref"] == clean_side else "other"
                rows.append({"problem_id": pid, "category": cat, "variant": variant, "mechanism": "judge",
                             "order": order, "pref_for_clean": pref, "parsed": rec["parsed"], "raw": rec["raw"],
                             "swapped": rec["swapped"], "outcome": outcome(pref, cat)})
    rows_path = run["outdir"] / "diagnostic_pairs.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"verifier_matches_expected_reward": verifier_ok})
    reconcile(run["outdir"], {"diagnostic_pairs.jsonl", "summary.json"})
    for mech in ("verifier", "judge"):
        s = summary[mech]
        print(f"{mech}: S_reason={s['S_reason']:.3f} S_outcome={s['S_outcome']:.3f} pooled={s['S_outcome_pooled']:.3f} "
              + " ".join(f"{c}=b{s[c]['better']:.2f}/t{s[c]['tie']:.2f}/w{s[c]['wrong']:.2f}"
                         for c in dict.fromkeys(CATEGORIES.values())))
    print(f"judge position consistency={summary['judge']['position_consistency']:.3f} "
          f"unparsed={summary['judge']['n_unparsed']}")
    print(f"{'PASS' if verifier_ok else 'BREACH'}: the verifier reproduces every response's expected exact reward")
    if not verifier_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

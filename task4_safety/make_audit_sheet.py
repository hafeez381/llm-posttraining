from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.guards import assert_audit_blind, sha256_file
from common.runmode import add_mode_arg

LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser(description="Build the blind manual-audit sheet, or lock it once every row is labelled")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--lock", action="store_true", help="lock the completed manual labels")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / ("_smoke/" if args.mode == "smoke" else "") / "task4_safety"
    outdir = base / "audit"
    sheet, lock = outdir / "manual_audit_sheet.csv", outdir / "audit_lock.json"

    if args.lock:
        labels = pd.read_csv(sheet, dtype=str, keep_default_na=False)
        bad = sorted(set(labels["manual_label"].str.strip()) - set(LABELS) - {""})
        if bad:
            raise SystemExit(f"Unknown labels in the sheet: {bad}; use one of {LABELS}")
        lock.write_text(json.dumps({"sha256": sha256_file(sheet), "n_rows": len(labels)}))
        assert_audit_blind(sheet, lock)
        print(f"Locked {len(labels)} manual labels, sha256 {sha256_file(sheet)}")
        return

    if sheet.exists() and (pd.read_csv(sheet, dtype=str, keep_default_na=False)["manual_label"].str.strip() != "").any():
        raise SystemExit(f"{sheet} already has manual labels; refusing to overwrite it")
    gen = read_jsonl(base / "generation" / "generated.jsonl")
    per_class = 2 if args.mode == "smoke" else int(cfg["manual_audit_per_class"])
    ids = fixed_audit_ids([r for r in gen if r["policy"] == "sft"], per_class, int(cfg["seed"]))
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xstest_id": ids}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    rows = pd.DataFrame([r for r in gen if r["xstest_id"] in set(ids)])
    rows = rows.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    rows.insert(0, "audit_row_id", range(1, len(rows) + 1))
    rows[["audit_row_id", "policy", "xstest_id"]].to_csv(outdir / "manual_audit_key.csv", index=False)
    rows = rows[["audit_row_id", "xstest_id", "prompt", "benchmark_class", "type", "response"]].assign(manual_label="")
    rows.to_csv(sheet, index=False)
    print(f"Wrote {len(rows)} rows to {sheet} ({len(ids)} prompts x {len(rows) // max(len(ids), 1)} policies)")
    print("Audit coverage by XSTest type:", rows.drop_duplicates("xstest_id")["type"].value_counts().to_dict())
    print(f"Label every row with one of {LABELS} without opening the judge files or manual_audit_key.csv, "
          "then run with --lock.")


if __name__ == "__main__":
    main()

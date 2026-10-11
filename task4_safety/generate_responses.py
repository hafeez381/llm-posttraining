from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.guards import assert_configs_frozen, assert_fixed_files, assert_standard_policies
from common.models import clear_gpu, load_policy, load_tokenizer
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import add_mode_arg, resolve_models

POLICIES = ("sft", "dpo", "ppo", "grpo")
DETERMINISM_ROWS = 12


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, df: pd.DataFrame | None = None):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg) if df is None else df
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def summarize(rows: list[dict]) -> dict:
    out = {}
    for p in dict.fromkeys(r["policy"] for r in rows):
        n = np.array([r["response_tokens"] for r in rows if r["policy"] == p], dtype=float)
        out[p] = {"n": len(n), "length_mean": float(n.mean()), "length_sd": float(n.std(ddof=1)),
                  "length_iqr": float(np.subtract(*np.percentile(n, [75, 25])))}
    return out


def main():
    ap = argparse.ArgumentParser(description="One greedy response per XSTest prompt for the four standard policies")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = resolve_models(load_yaml(args.config), args.mode)
    smoke = args.mode == "smoke"
    assert_fixed_files(args.mode)
    df = load_xstest(cfg)
    df = df.groupby("benchmark_class", sort=False).head(4) if smoke else df
    frozen_sha = None
    if smoke:
        cfg["policies"].update(dpo=None, ppo=None, grpo=None)
        cfg["safety_max_new_tokens"] = 16
    else:
        frozen_sha = assert_configs_frozen()
        assert_standard_policies(policy_specs(cfg))
    run = start_run(f"results/{'_smoke/' if smoke else ''}task4_safety/generation", mode=args.mode, cfg=cfg,
                    seed=int(cfg["seed"]))

    rows, deterministic = [], None
    for name in POLICIES:
        recs = generate_for_policy(cfg, name, df=df)
        if name == "sft":
            again = generate_for_policy(cfg, name, df=df.iloc[:DETERMINISM_ROWS])
            deterministic = all(a["response"] == b["response"] for a, b in zip(recs, again))
        rows += recs
        clear_gpu()
        print(f"{name}: {len(recs)} responses, mean tokens {np.mean([r['response_tokens'] for r in recs]):.1f}",
              flush=True)

    rows_path = run["outdir"] / "generated.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    finalize(run, rows_path, summary, summarize, extra={"frozen_at": frozen_sha, "deterministic_rerun": deterministic,
                                                         "decoding": "greedy, batch 4, prompt cap 256, "
                                                                     f"{cfg['safety_max_new_tokens']} new tokens"})
    reconcile(run["outdir"], {"generated.jsonl", "summary.json"})
    order = {p: [r["xstest_id"] for r in rows if r["policy"] == p] for p in POLICIES}
    same_order = all(order[p] == order["sft"] for p in POLICIES)
    for ok, what in ((same_order, "every policy answers the same prompts in the same order"),
                     (deterministic, "re-generating SFT rows reproduces them byte for byte")):
        print(f"{'PASS' if ok else 'BREACH'}: {what}")
    if not (same_order and deterministic):
        sys.exit(1)


if __name__ == "__main__":
    main()

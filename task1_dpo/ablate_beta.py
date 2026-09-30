from __future__ import annotations

import argparse
import sys

from common.data import load_yaml, read_jsonl
from common.guards import GuardError, assert_matched_budget
from common.runmode import add_mode_arg
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def fork_name(beta: float) -> str:
    return f"beta_{beta:g}"


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="evaluate existing fork adapters only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    prefix = "_smoke/" if args.mode == "smoke" else ""

    runs = []
    for beta in [float(b) for b in cfg["betas"]]:
        name = fork_name(beta)
        adapter = f"outputs/{prefix}task1_dpo/{name}"
        if not args.skip_train:
            run_training(args.config, name, output_path=adapter, beta=beta,
                         max_examples=int(cfg["short_ablation_examples"]), mode=args.mode)
        train = read_jsonl(f"results/{prefix}task1_dpo/{name}/train/train_log.jsonl")
        evaluate(args.config, adapter, name, beta=beta, mode=args.mode,
                 budget=f"{cfg['short_ablation_examples']} pairs, {len(train)} steps (short fork)")
        runs.append({"beta": beta, "updates": len(train), "k": int(cfg["batch_size"]),
                     "cap": int(cfg["max_sequence_length"]), "seed": int(cfg["seed"]),
                     "prompt_ids": [p for r in train for p in r["prompt_ids"]],
                     "margin_by_step": [r["margin_mean"] for r in train]})

    try:
        assert_matched_budget(runs)
    except GuardError as e:
        print(f"BREACH: {e}")
        sys.exit(1)
    print("PASS: forks share update count, data order, batch size, cap and seed")
    last = {r["beta"]: r["margin_by_step"][-1] for r in runs}
    ordered = [last[b] for b in sorted(last)]
    print("final train margin by beta:", {b: round(m, 4) for b, m in sorted(last.items())},
          "(larger beta -> smaller margin expected:", all(a >= b for a, b in zip(ordered, ordered[1:])), ")")


if __name__ == "__main__":
    main()

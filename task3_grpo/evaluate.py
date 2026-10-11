from __future__ import annotations

import argparse

from common.runmode import add_mode_arg
from task2_ppo.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, task="task3_grpo", mode=args.mode)


if __name__ == "__main__":
    main()

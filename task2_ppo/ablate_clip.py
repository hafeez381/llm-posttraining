from __future__ import annotations

import argparse

from common.runmode import add_mode_arg
from task2_ppo.ablate_kl import run_study


def main():
    ap = argparse.ArgumentParser(description="Clipping forks from the PPO midpoint")
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    run_study("clip", args.config, args.mode)


if __name__ == "__main__":
    main()

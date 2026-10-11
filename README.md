# ATML PA2: LLM Post-Training

DPO, PPO and GRPO post-training of Qwen2.5-1.5B-Instruct with LoRA, a safety-calibration study on XSTest, and an RLVR vs RLAIF feedback-source comparison. Built on the course starter repository; every number in the report traces to a file under `results/`, written by a committed script with a recorded command, config, seed and git sha.

## Setup

```bash
python -m pip install -r requirements.txt     # pinned versions from the course release
python -m scripts.download_assets             # fixed data, cached rollouts, supplied checkpoints
python -m scripts.validate_assets
```

Every script takes `--mode {smoke,full}`. `smoke` swaps every model for a tiny test model and runs a few steps on CPU; `full` is the real run (the GPU runs used one Colab L4 for Tasks 1–3 and one T4 for Tasks 4–5). `python -m pytest tests -q` runs the unit tests, including one failing-first test per starter fix.

## Reproducing the results

Run from the repo root, in this order (later tasks read earlier outputs). Each command writes `results/<task>/<run>/` with per-item rows and a `summary.json` recomputed from them.

### Shared checks
```bash
python -m task2_ppo.check_cache_logprobs --mode full --dtype bfloat16 --truncation right   # log-prob temperature, value alignment, ratio = 1
python -m common.eval_protocol --mode full                                                   # held-out evaluation reproduces itself
```

### Task 1: DPO
```bash
python -m task1_dpo.train --mode full --run-name standard
python -m task1_dpo.train --mode full --run-name length_balanced --dataset data/dpo_length_balanced_train.jsonl
python -m task1_dpo.evaluate --mode full --adapter none --name sft
python -m task1_dpo.evaluate --mode full --adapter outputs/task1_dpo/standard --name standard --budget "1500 pairs, 94 steps (one epoch)"
python -m task1_dpo.evaluate --mode full --adapter outputs/task1_dpo/length_balanced --name length_balanced --budget "1500 pairs, 94 steps (one epoch)"
python -m task1_dpo.ablate_beta --mode full          # beta in {0.03, 0.10, 0.30}, 600 pairs / 38 steps each
python -m task1_dpo.analyze_length --mode full       # length strata, word-limit prompts, qualitative candidates
```

### Task 2: PPO
```bash
python -m task2_ppo.continue_train --mode full --run-name standard       # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate --mode full --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --mode full --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.ablate_kl --mode full            # beta_KL in {0, 0.10, 0.20}, 8 updates each
python -m task2_ppo.ablate_clip --mode full          # epsilon in {0.05, 0.20, 0.50}; reuses the shared 0.20 / 0.10 fork
python -m task2_ppo.analyze_clipping --mode full     # clipped surrogate on the 32 cached rollouts
python -m task2_ppo.analyze_overopt --mode full      # reward vs quality candidates
```

### Task 3: GRPO
```bash
python -m task3_grpo.analyze_group_size --mode full                       # K in {2, 4, 8} at equal total generations (CPU)
python -m task3_grpo.continue_train --mode full --run-name standard       # 20 updates from the supplied midpoint
python -m task3_grpo.evaluate --mode full --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --mode full --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.compare_normalization --mode full   # canonical vs Dr. GRPO, 8 updates each
python -m task3_grpo.length_gradients --mode full        # per-completion gradient norm vs length on the fixed cache
```

`results/frozen_policies.json` records the standard DPO, PPO and GRPO adapters and their sha256. It was committed before any Task 4 or Task 5 run, and both tasks check it.

### Task 4: Safety calibration
```bash
python -m task4_safety.generate_responses --mode full    # greedy, the four frozen policies, all 450 XSTest prompts
python -m task4_safety.judge_responses --mode full       # fixed AI judge, raw output kept
python -m task4_safety.make_audit_sheet --mode full      # blind, shuffled sheet: 60 prompts x 4 policies
# ... label results/task4_safety/audit/manual_audit_sheet.csv by hand, then:
python -m task4_safety.make_audit_sheet --mode full --lock
python -m task4_safety.evaluate_safety --mode full       # rates, category labels, judge vs manual agreement
```

### Task 5: RLVR vs RLAIF
```bash
python -m task5_feedback.evaluate_math --mode full --dataset gsm
python -m task5_feedback.evaluate_math --mode full --dataset transfer
python -m task5_feedback.judge_pairwise --mode full      # win rate vs SFT, verifier-judge agreement
python -m task5_feedback.score_perturbations --mode full # controlled diagnostics, judge in both orders
python -m task5_feedback.compare_feedback --mode full
```

### Report tables and figures
```bash
for t in 1 2 3 4 5; do python report/build_task${t}_artifacts.py; done
```

## Fixes to the starter code

Each is its own `fix(...)` commit with a unit test that fails on the original code.

| Commit | Fix |
|---|---|
| `bb89e89` | DPO: subtract (not add) the reference log-ratio in the logit |
| `530deca` | PPO: take the minimum (not maximum) in the clipped surrogate |
| `c704ae2` | GRPO: normalize advantages within each prompt group, not over the batch |
| `6719634` | Generation: left-truncate long prompts so the assistant header survives |
| `653def4` | Reward model: left-truncate so the response, not the prompt, is scored |
| `e60ec0a` | Log-probs: divide logits by the sampling temperature |
| `8b882bd` | Log-prob forwards: dropout off, so the PPO/GRPO ratio starts at exactly 1 |
| `616d5c4` | Evaluation: chunked log-prob forwards so full-vocabulary logits fit in memory |
| `b7fc96c` | PPO critic: read values from the backbone, not the classifier |
| `3c99aa5` | Pairwise judge: keep its raw output so ties and parse failures can be told apart |

## Attribution

- **Course starter** (ATML PA2, https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining) and **course assets** (https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets): model and data loaders, objective helpers, the reward model, the AI judges and their prompts, the exact verifier, and the fixed datasets, caches and midpoint checkpoints. Used as distributed for the course; files keep the starter's layout, and the fixes above are listed separately.
- **TRL** (https://github.com/huggingface/trl, Apache-2.0): the practice of disabling dropout for RL log-prob consistency (`common/rl_utils.disable_dropout`) and the `prev` value-position convention. Ideas only; no code copied.
- **Dr. GRPO** (Liu et al., 2025, https://github.com/sail-sg/understand-r1-zero): the length-normalization comparison follows the paper; the course release supplies the `dr_grpo` loss used here.
- **KL estimators** (J. Schulman, "Approximating KL Divergence", http://joschu.net/blog/kl-approx.html): the k1/k3 estimators in `common/rl_utils.py`.
- Models: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct (Qwen license), the course reward model `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`.

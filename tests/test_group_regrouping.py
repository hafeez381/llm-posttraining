import math

import torch

from task3_grpo.analyze_group_size import (difficulty_bins, exhaustive_subset_stats, load_k8_cache,
                                           regroup_equal_generation_budget)
from task3_grpo.grpo import group_stats


def _cache():
    return load_k8_cache("cached/grpo_k_cache.jsonl")


def test_equal_generations_and_no_mixed_groups():
    by_prompt = _cache()
    for k, n_groups in [(2, 96), (4, 48), (8, 24)]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        assert len(groups) == n_groups
        assert sum(len(g["rewards"]) for g in groups) == 192
        for g in groups:
            prompt_rows = by_prompt[g["source_index"]]
            assert g["rewards"] == [float(prompt_rows[i]["reward"]) for i in g["gen_indices"]]


def test_exhaustive_subsets_hand_computed():
    s = exhaustive_subset_stats([1, 1, 1, 1, 1, 1, 2, 2], k=2)
    assert s["n_subsets"] == math.comb(8, 2)
    assert math.isclose(s["informative_frac"], 12 / 28)
    assert math.isclose(s["mean_std"], 12 / 28 * 0.5)


def test_bins_are_equal_thirds_ordered_by_mean_reward():
    by_prompt = _cache()
    bins, _ = difficulty_bins(by_prompt)
    mean = {p: sum(r["reward"] for r in rows) / 8 for p, rows in by_prompt.items()}
    counts = {b: sum(v == b for v in bins.values()) for b in ("hard", "medium", "easy")}
    assert counts == {"hard": 8, "medium": 8, "easy": 8}
    assert max(mean[p] for p in bins if bins[p] == "hard") <= min(mean[p] for p in bins if bins[p] == "medium")
    assert max(mean[p] for p in bins if bins[p] == "medium") <= min(mean[p] for p in bins if bins[p] == "easy")


def test_group_stats_tolerance():
    stats = group_stats(torch.tensor([2.0, 2.0, 1.0, 3.0]), torch.tensor([0, 0, 1, 1]))
    assert [s["informative"] for s in stats] == [False, True]
    assert [s["informative_isclose"] for s in stats] == [False, True]
    assert math.isclose(stats[1]["std"], 1.0)

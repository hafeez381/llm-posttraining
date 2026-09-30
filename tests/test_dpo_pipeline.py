import math

import task1_dpo.train as T
from common.data import read_jsonl
from task1_dpo.evaluate import summarize_pairs, wilson


def test_wilson_hand_computed():
    lo, hi = wilson(41, 82)
    assert math.isclose(lo, 0.3943, abs_tol=1e-4) and math.isclose(hi, 0.6057, abs_tol=1e-4)


def test_pair_accuracy_is_strict():
    rows = [{"margin": m, "loss_at_beta": 0.0, "beta": 0.1, "truncated_c": False, "truncated_r": False}
            for m in (0.0, 0.0, 1.0, -1.0)]
    assert summarize_pairs(rows)["accuracy"] == 0.25


def test_resume_reproduces_an_uninterrupted_run(monkeypatch):
    full = T.run_training("configs/dpo.yaml", "test_resume_ref", mode="smoke")
    orig, calls = T._save_snapshot, []

    def stop_after_first(*a):
        orig(*a)
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(T, "_save_snapshot", stop_after_first)
    try:
        T.run_training("configs/dpo.yaml", "test_resume", mode="smoke")
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(T, "_save_snapshot", orig)
    resumed = T.run_training("configs/dpo.yaml", "test_resume", mode="smoke", resume=True)
    assert resumed["adapter_sha256"] == full["adapter_sha256"]
    a = read_jsonl("results/_smoke/task1_dpo/test_resume_ref/train/train_log.jsonl")
    b = read_jsonl("results/_smoke/task1_dpo/test_resume/train/train_log.jsonl")
    assert [r["loss"] for r in a] == [r["loss"] for r in b]

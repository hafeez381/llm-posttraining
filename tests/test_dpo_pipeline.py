import math

import task1_dpo.train as T
from common.data import read_jsonl


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

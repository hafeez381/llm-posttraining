import json

import pytest

from common.results import finalize, reconcile, start_run, write_rows


def summarize(rows):
    return {"n": len(rows), "mean": sum(r["x"] for r in rows) / len(rows)}


def test_start_run_refuses_paths_outside_results_and_outputs():
    for bad in ["data", "results", "results/../common", "/tmp/x"]:
        with pytest.raises(ValueError):
            start_run(bad, mode="smoke", cfg={}, seed=0)


def test_finalize_recomputes_from_rows_and_catches_tampering():
    run = start_run("results/_smoke/test_results", mode="smoke", cfg={"a": 1}, seed=0)
    rows = [{"x": 1.0}, {"x": 2.0}]
    rows_path = run["outdir"] / "rows.jsonl"
    write_rows(rows_path, rows)
    out = finalize(run, rows_path, summarize(rows), summarize)
    record = json.loads(out.read_text())
    assert record["summary"] == {"n": 2, "mean": 1.5} and record["n_rows"] == 2
    assert {"git_sha", "mode", "seed", "config", "command", "wall_clock_s", "peak_vram_gb"} <= record.keys()
    reconcile(run["outdir"], {"rows.jsonl", "summary.json"})

    write_rows(rows_path, [{"x": 1.0}, {"x": 5.0}])
    with pytest.raises(AssertionError, match="does not match"):
        finalize(run, rows_path, summarize(rows), summarize)


def test_start_run_clears_and_reconcile_checks_both_ways():
    run = start_run("results/_smoke/test_reconcile", mode="smoke", cfg={}, seed=0)
    (run["outdir"] / "stale.json").write_text("{}")
    run = start_run("results/_smoke/test_reconcile", mode="smoke", cfg={}, seed=0)
    assert not any(run["outdir"].iterdir())
    (run["outdir"] / "a.json").write_text("{}")
    with pytest.raises(AssertionError, match="missing"):
        reconcile(run["outdir"], {"a.json", "b.json"})
    (run["outdir"] / "b.json").write_text("{}")
    (run["outdir"] / "extra.json").write_text("{}")
    with pytest.raises(AssertionError, match="unexpected"):
        reconcile(run["outdir"], {"a.json", "b.json"})

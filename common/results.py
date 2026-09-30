"""Result writing: per-item rows first, aggregates recomputed from the persisted rows, and a provenance stamp."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common.data import REPO_ROOT, read_jsonl, repo_path, write_jsonl

WRITABLE_ROOTS = ("results", "outputs")


def git_sha() -> dict:
    """HEAD and a dirty flag locally; the shipped GIT_SHA file on the Colab VM (no .git there)."""
    if (REPO_ROOT / ".git").exists():
        run = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    sha_file = REPO_ROOT / "GIT_SHA"
    return {"git_sha": sha_file.read_text().strip() if sha_file.exists() else "unknown", "git_dirty": False}


def start_run(outdir: str | Path, *, mode: str, cfg: dict, seed: int, resume: bool = False) -> dict:
    """Empty `outdir` (only under results/ or outputs/) and return the run context with its stamp."""
    outdir = repo_path(outdir).resolve()
    if not any(outdir.is_relative_to(REPO_ROOT / r) and outdir != REPO_ROOT / r for r in WRITABLE_ROOTS):
        raise ValueError(f"Refusing to clear {outdir}: runs write only below results/ or outputs/")
    if resume:
        if not outdir.is_dir():
            raise FileNotFoundError(f"Cannot resume: {outdir} does not exist")
    else:
        shutil.rmtree(outdir, ignore_errors=True)
        outdir.mkdir(parents=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp = {
        **git_sha(),
        "mode": mode,
        "seed": seed,
        "command": " ".join(["python", *sys.orig_argv[1:]]),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
    }
    return {"outdir": outdir, "stamp": stamp, "t0": time.perf_counter()}


def write_rows(path: str | Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)


def _same(a, b, rel=1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


def finalize(run: dict, rows_path: str | Path, summary: dict, summarize_fn, name: str = "summary.json",
             extra: dict | None = None) -> Path:
    """Recompute the summary from the rows on disk, check it matches, and write it with the stamp."""
    rows_path = repo_path(rows_path)
    recomputed = summarize_fn(read_jsonl(rows_path))
    if not _same(summary, recomputed):
        raise AssertionError(f"Summary does not match the rows in {rows_path}")
    out = run["outdir"] / name
    record = {
        **run["stamp"],
        "wall_clock_s": round(time.perf_counter() - run["t0"], 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,
        "rows": str(rows_path.relative_to(REPO_ROOT)),
        "n_rows": len(read_jsonl(rows_path)),
        **(extra or {}),
        "summary": summary,
    }
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def reconcile(outdir: str | Path, expected: set[str]) -> None:
    """Every expected file exists, and every file in `outdir` is expected (paths relative to `outdir`)."""
    outdir = repo_path(outdir)
    found = {str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()}
    missing, extra = sorted(set(expected) - found), sorted(found - set(expected))
    if missing or extra:
        raise AssertionError(f"Reconcile failed in {outdir}: missing={missing} unexpected={extra}")

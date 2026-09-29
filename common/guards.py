"""Runtime integrity guards.

Each guard raises instead of warning: a violation invalidates a whole task, so it has to stop the run.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from common.data import REPO_ROOT, read_jsonl

MANIFEST = "manifests/sha256.json"
WORD_LIMIT_PROMPTS = "data/word_limit_prompts.jsonl"
# Git blob id of data/word_limit_prompts.jsonl in the course starter (commit 05543be).
WORD_LIMIT_BLOB = "9a0692f7c8e1b05c3a96104c70c46840dd45794e"
FROZEN_POLICIES = "results/frozen_policies.json"
STANDARD_POLICIES = {"sft", "dpo", "ppo", "grpo"}
FORK_MARKERS = ("beta_", "eps_", "kl_", "dr_grpo", "length_balanced")
FROZEN_PATHS = ["configs", "task1_dpo", "task2_ppo", "task3_grpo", "common"]


class GuardError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha(path: str | Path) -> str:
    """Same id as `git hash-object`, computed without git (the Colab VM has no .git)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))


def assert_fixed_files(mode: str = "full", root: str | Path = REPO_ROOT, entries: dict | None = None) -> list[str]:
    """Check every released file against the course sha256 manifest.

    The manifest's own entry is skipped: a file cannot contain its own hash. A missing file is fatal in
    full mode; smoke mode only reports it, since the 1 GB critic weights live on the GPU VM only.
    Returns the missing paths.
    """
    root = Path(root)
    entries = _manifest(root) if entries is None else entries
    missing, changed = [], []
    for rel, expected in sorted(entries.items()):
        if rel == MANIFEST:
            continue
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != expected:
            changed.append(rel)
    if git_blob_sha(root / WORD_LIMIT_PROMPTS) != WORD_LIMIT_BLOB:
        changed.append(WORD_LIMIT_PROMPTS)
    if changed:
        raise GuardError("Fixed course files changed: " + ", ".join(changed))
    if missing and mode == "full":
        raise GuardError("Fixed course files missing: " + ", ".join(missing))
    if missing:
        print(f"[guards] smoke: {len(missing)} fixed file(s) absent locally: {', '.join(missing)}")
    return missing


def assert_same_midpoint(kind: str, adapter_dir: str | Path, value_dir: str | Path | None = None,
                         root: str | Path = REPO_ROOT) -> None:
    """Every PPO/GRPO run and fork must start from the supplied midpoint, byte for byte."""
    root = Path(root)
    entries = _manifest(root)
    checks = [(Path(adapter_dir) / "adapter_model.safetensors",
               f"checkpoints/{kind}_midpoint_policy/adapter_model.safetensors")]
    if kind == "ppo":
        if value_dir is None:
            raise GuardError("PPO runs must also pass the critic checkpoint")
        checks.append((Path(value_dir) / "model.safetensors", "checkpoints/ppo_midpoint_value/model.safetensors"))
    for path, key in checks:
        path = path if path.is_absolute() else root / path
        if sha256_file(path) != entries[key]:
            raise GuardError(f"{path} is not the supplied {key}")


def assert_train_split(prompt_ids=None, eval_path: str | Path | None = None, opened_path: str | Path | None = None,
                       id_field: str = "prompt_id") -> None:
    """Training never sees held-out prompts.

    `prompt_ids` are checked against the ids in `eval_path`; `opened_path` is a data file a training loop
    is about to read, which must not be a DPO eval split. Match on `prompt_id`: `source_index` is only
    unique within a source split, and 3 values occur in both RL prompt pools.
    """
    if opened_path is not None:
        name = Path(opened_path).name
        if name.startswith("dpo_") and "_eval" in name:
            raise GuardError(f"A training loop opened the held-out file {name}")
    if prompt_ids is not None:
        held_out = {str(r[id_field]) for r in read_jsonl(eval_path)}
        leaked = sorted({str(i) for i in prompt_ids} & held_out)
        if leaked:
            raise GuardError(f"{len(leaked)} training prompt(s) are held-out: {leaked[:5]}")


def mark_analysis_only(policy):
    """Tag a throwaway policy (updated on the eval-prompt rollout cache) so it can never be saved."""
    policy._analysis_only = True
    return policy


def assert_analysis_only(policy) -> None:
    """Call before saving or evaluating a policy."""
    if getattr(policy, "_analysis_only", False):
        raise GuardError("This policy was updated on eval-prompt rollouts for analysis; it must not be saved or evaluated")


def assert_configs_frozen(root: str | Path = REPO_ROOT) -> str:
    """Task 4/5 evaluation runs only after the Task 1-3 code, configs and standard adapters are committed.

    Returns the commit sha to record. On the Colab VM the code arrives as `git archive HEAD`, which is
    clean by construction, so the shipped GIT_SHA file is the record there.
    """
    root = Path(root)
    if not (root / FROZEN_POLICIES).exists():
        raise GuardError(f"{FROZEN_POLICIES} is missing: freeze the standard policies first")
    if (root / ".git").exists():
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *FROZEN_PATHS], cwd=root)
        if r.returncode != 0:
            raise GuardError("Uncommitted changes under " + " ".join(FROZEN_PATHS))
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    sha_file = root / "GIT_SHA"
    if not sha_file.exists():
        raise GuardError("No .git and no GIT_SHA file: cannot tie this run to a commit")
    return sha_file.read_text().strip()


def assert_standard_policies(specs: dict[str, str], root: str | Path = REPO_ROOT) -> None:
    """Task 4 evaluates exactly SFT, standard DPO, standard PPO and standard GRPO, never an ablation fork.

    `specs` maps policy name to adapter directory (None for SFT).
    """
    root = Path(root)
    frozen = json.loads((root / FROZEN_POLICIES).read_text(encoding="utf-8"))
    if set(specs) != STANDARD_POLICIES:
        raise GuardError(f"Expected policies {sorted(STANDARD_POLICIES)}, got {sorted(specs)}")
    for name, path in specs.items():
        if path is None:
            if name != "sft":
                raise GuardError(f"{name} has no adapter")
            continue
        if any(m in str(path) for m in FORK_MARKERS):
            raise GuardError(f"{name} points at an ablation fork: {path}")
        rec = frozen[name]
        adapter = Path(path) / "adapter_model.safetensors"
        adapter = adapter if adapter.is_absolute() else root / adapter
        if str(path) != rec["path"] or sha256_file(adapter) != rec["sha256"]:
            raise GuardError(f"{name} adapter differs from the frozen record")


def assert_audit_blind(labels_csv: str | Path, lock_json: str | Path) -> None:
    """Manual labels are complete and locked before any AI-vs-manual join."""
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    empty = int((labels["manual_label"].str.strip() == "").sum())
    if empty:
        raise GuardError(f"{empty} manual label(s) still empty")
    lock_json = Path(lock_json)
    if not lock_json.exists():
        raise GuardError("Manual labels are not locked (no audit_lock.json)")
    if json.loads(lock_json.read_text())["sha256"] != sha256_file(labels_csv):
        raise GuardError("Manual labels changed after they were locked")


def sequence_hash(ids) -> str:
    return hashlib.sha256(json.dumps([str(i) for i in ids]).encode()).hexdigest()


def assert_matched_budget(runs: list[dict], keys=("updates", "k", "cap", "seed")) -> None:
    """Forks within one study use the same update count, prompt sequence, group size, cap and seed."""
    ref = runs[0]
    for run in runs[1:]:
        for key in keys:
            if run[key] != ref[key]:
                raise GuardError(f"Budget mismatch on {key}: {ref[key]} vs {run[key]}")
        if sequence_hash(run["prompt_ids"]) != sequence_hash(ref["prompt_ids"]):
            raise GuardError("Prompt-id sequences differ between forks")

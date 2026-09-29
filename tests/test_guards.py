import json
import shutil
import urllib.request

import pandas as pd
import pytest
import torch.nn as nn

from common import guards
from common.data import REPO_ROOT, read_jsonl
from common.guards import GuardError

GSM = "data/gsm8k_eval.jsonl"


def _manifest():
    return json.loads((REPO_ROOT / guards.MANIFEST).read_text())


def _copy(tmp_path, rels):
    for rel in rels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)


def test_fixed_files_pass_on_released_assets():
    missing = guards.assert_fixed_files("smoke")
    assert set(missing) <= {"checkpoints/ppo_midpoint_value/model.safetensors"}


def test_fixed_files_catch_a_flipped_byte_in_a_copy(tmp_path):
    _copy(tmp_path, [GSM, guards.WORD_LIMIT_PROMPTS])
    entries = {GSM: _manifest()[GSM]}
    guards.assert_fixed_files("full", root=tmp_path, entries=entries)
    data = bytearray((tmp_path / GSM).read_bytes())
    data[100] ^= 1
    (tmp_path / GSM).write_bytes(bytes(data))
    with pytest.raises(GuardError, match="changed"):
        guards.assert_fixed_files("full", root=tmp_path, entries=entries)


def test_fixed_files_missing_is_fatal_only_in_full(tmp_path):
    _copy(tmp_path, [guards.WORD_LIMIT_PROMPTS])
    entries = {GSM: _manifest()[GSM]}
    assert guards.assert_fixed_files("smoke", root=tmp_path, entries=entries) == [GSM]
    with pytest.raises(GuardError, match="missing"):
        guards.assert_fixed_files("full", root=tmp_path, entries=entries)


def test_word_limit_prompts_match_the_starter_blob(tmp_path):
    _copy(tmp_path, [guards.WORD_LIMIT_PROMPTS])
    guards.assert_fixed_files("full", root=tmp_path, entries={})
    (tmp_path / guards.WORD_LIMIT_PROMPTS).write_text("{}\n")
    with pytest.raises(GuardError, match="word_limit"):
        guards.assert_fixed_files("full", root=tmp_path, entries={})


def test_transfer_file_equals_a_fresh_svamp_rebuild():
    """One-time check that the bundled SVAMP file is the course subset; its manifest sha pins it afterwards."""
    from scripts.prepare_transfer_eval import SVAMP_URL, build_transfer_rows
    try:
        with urllib.request.urlopen(SVAMP_URL, timeout=30) as r:
            records = json.loads(r.read().decode("utf-8"))
    except OSError:
        pytest.skip("SVAMP source unreachable")
    assert build_transfer_rows(records) == read_jsonl("data/math_transfer_eval.jsonl")


def test_same_midpoint():
    guards.assert_same_midpoint("grpo", "checkpoints/grpo_midpoint_policy")
    with pytest.raises(GuardError):
        guards.assert_same_midpoint("grpo", "checkpoints/grpo_midpoint_policy_v3_masked512")
    with pytest.raises(GuardError, match="critic"):
        guards.assert_same_midpoint("ppo", "checkpoints/ppo_midpoint_policy")


def test_train_split():
    train = [r["prompt_id"] for r in read_jsonl("data/rl_prompt_pool_train.jsonl")]
    held_out = read_jsonl("data/rl_prompt_pool_eval.jsonl")[0]["prompt_id"]
    guards.assert_train_split(train, "data/rl_prompt_pool_eval.jsonl")
    with pytest.raises(GuardError, match="held-out"):
        guards.assert_train_split(train[:3] + [held_out], "data/rl_prompt_pool_eval.jsonl")
    guards.assert_train_split(opened_path="data/dpo_standard_train.jsonl")
    with pytest.raises(GuardError):
        guards.assert_train_split(opened_path="data/dpo_standard_eval.jsonl")


def test_analysis_only():
    policy = nn.Linear(2, 2)
    guards.assert_analysis_only(policy)
    with pytest.raises(GuardError):
        guards.assert_analysis_only(guards.mark_analysis_only(policy))


def test_configs_frozen_needs_the_frozen_record(tmp_path):
    with pytest.raises(GuardError, match="frozen_policies"):
        guards.assert_configs_frozen(root=tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / guards.FROZEN_POLICIES).write_text("{}")
    with pytest.raises(GuardError, match="GIT_SHA"):
        guards.assert_configs_frozen(root=tmp_path)
    (tmp_path / "GIT_SHA").write_text("abc123\n")
    assert guards.assert_configs_frozen(root=tmp_path) == "abc123"


def test_standard_policies(tmp_path):
    frozen = {}
    specs = {"sft": None}
    for name in ["dpo", "ppo", "grpo"]:
        d = tmp_path / "outputs" / name
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(name.encode())
        specs[name] = str(d)
        frozen[name] = {"path": str(d), "sha256": guards.sha256_file(d / "adapter_model.safetensors")}
    (tmp_path / "results").mkdir()
    (tmp_path / guards.FROZEN_POLICIES).write_text(json.dumps(frozen))
    guards.assert_standard_policies(specs, root=tmp_path)
    with pytest.raises(GuardError, match="fork"):
        guards.assert_standard_policies({**specs, "dpo": str(tmp_path / "outputs/beta_0.3")}, root=tmp_path)
    with pytest.raises(GuardError, match="Expected policies"):
        guards.assert_standard_policies({k: v for k, v in specs.items() if k != "grpo"}, root=tmp_path)
    (tmp_path / "outputs/ppo/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(GuardError, match="differs"):
        guards.assert_standard_policies(specs, root=tmp_path)


def test_audit_blind(tmp_path):
    labels, lock = tmp_path / "labels.csv", tmp_path / "audit_lock.json"
    pd.DataFrame({"xstest_id": [1, 2], "manual_label": ["SAFE_ANSWER", ""]}).to_csv(labels, index=False)
    with pytest.raises(GuardError, match="empty"):
        guards.assert_audit_blind(labels, lock)
    pd.DataFrame({"xstest_id": [1, 2], "manual_label": ["SAFE_ANSWER", "OVER_REFUSAL"]}).to_csv(labels, index=False)
    with pytest.raises(GuardError, match="not locked"):
        guards.assert_audit_blind(labels, lock)
    lock.write_text(json.dumps({"sha256": guards.sha256_file(labels)}))
    guards.assert_audit_blind(labels, lock)
    labels.write_text(labels.read_text().replace("OVER_REFUSAL", "SAFE_ANSWER"))
    with pytest.raises(GuardError, match="changed"):
        guards.assert_audit_blind(labels, lock)


def test_matched_budget():
    run = {"updates": 8, "k": 4, "cap": 512, "seed": 6304, "prompt_ids": ["a", "b"]}
    guards.assert_matched_budget([run, dict(run)])
    with pytest.raises(GuardError, match="updates"):
        guards.assert_matched_budget([run, {**run, "updates": 7}])
    with pytest.raises(GuardError, match="sequences"):
        guards.assert_matched_budget([run, {**run, "prompt_ids": ["b", "a"]}])

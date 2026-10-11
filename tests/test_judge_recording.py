from common.data import load_yaml
from common.runmode import resolve_models
from task5_feedback.judge_recording import RecordingPairwiseJudge
from task5_feedback.rlaif import PairwiseAIJudge


def test_recording_judge_matches_released_judge(tmp_path):
    cfg = resolve_models(load_yaml("configs/feedback.yaml"), "smoke")
    base = PairwiseAIJudge(cfg, tmp_path / "base.json")
    rec = RecordingPairwiseJudge(cfg, tmp_path / "rec.json", tmp_path / "side.json")
    pairs = [(f"What is {i}+{i}?", f"#### {2 * i}", f"#### {2 * i + 1}") for i in range(20)]
    for p, a, b in pairs:
        out = rec.compare_recorded(p, a, b)
        assert out["pref"] == base.compare(p, a, b)
        assert out["swapped"] == (int(rec._key(p, a, b)[:8], 16) % 2 == 1)
    again = RecordingPairwiseJudge(cfg, tmp_path / "rec.json", tmp_path / "side.json")
    assert again.compare_recorded(*pairs[0]) == rec.compare_recorded(*pairs[0])

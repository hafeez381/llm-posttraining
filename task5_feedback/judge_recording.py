from __future__ import annotations

import json
import re

from common.data import repo_path
from task5_feedback.rlaif import PairwiseAIJudge


class RecordingPairwiseJudge(PairwiseAIJudge):
    def __init__(self, cfg: dict, cache_path, sidecar_path):
        super().__init__(cfg, cache_path)
        self.sidecar_path = repo_path(sidecar_path)
        self.sidecar = json.loads(self.sidecar_path.read_text()) if self.sidecar_path.exists() else {}
        decode = self.tokenizer.decode
        def recording_decode(*args, **kwargs):
            self._last_raw = decode(*args, **kwargs)
            return self._last_raw
        self.tokenizer.decode = recording_decode

    def compare_recorded(self, problem: str, a: str, b: str) -> dict:
        key = self._key(problem, a, b)
        self._last_raw = None
        pref = self.compare(problem, a, b)
        if key not in self.sidecar:
            raw = self._last_raw or ""
            self.sidecar[key] = {"raw": raw, "parsed": bool(re.search(r"\b(A|B|TIE)\b", raw.strip().upper())),
                                 "swapped": int(key[:8], 16) % 2 == 1}
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            self.sidecar_path.write_text(json.dumps(self.sidecar))
        return {"pref": pref, **self.sidecar[key]}

"""Learning during use: what the operators on this line confirmed, on top of the rule table.

When the advice says "check the phototool at this spot", the operator can record what the cause turned out to be.
Each confirmation is a real, local label. The rule table (kb/evidence.yaml) stays the starting point; confirmations
move this line's numbers towards what actually happens here, the way a process engineer's experience would:

  share of a cause      (K_PRIOR x table share + confirmed count) / (K_PRIOR + confirmations of that type)
  chance of a clue      (K_CLUE x table chance + times seen with it) / (K_CLUE + times the clue was known)

So 20 confirmations of a defect type weigh as much as the table's own share, and one odd case moves little.
Llama's probabilities are then reweighted by how far this line's posterior has moved from the table's, so the
fine-tuned model keeps doing the reading and the line's experience shifts it, without retraining on the device.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from .causes import CauseModel, features_from

K_PRIOR = 20.0
K_CLUE = 10.0


class LineCauses:
    def __init__(self, cm: CauseModel, rows: list[tuple[str, str, dict]]):
        """rows: (defect type, confirmed cause id, the evidence shown with the advice)."""
        self.table = cm
        self.n_type: dict[str, int] = defaultdict(int)
        n_cause: dict[str, int] = defaultdict(int)
        seen: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # true, known
        for cls, cause, ev in rows:
            if cause not in cm.evidence:
                continue
            self.n_type[cls] += 1
            n_cause[cause] += 1
            feats = features_from(ev or {})
            if (ev or {}).get("roll_pitch") is False:   # passed to the model only when found, but its absence
                feats["roll_pitch"] = False             # on a confirmed board is real evidence for learning
            for k, v in feats.items():
                if k == "round" and features_from(ev).get("long_thin"):
                    continue                # round is only asked when the defect is not long and thin
                seen[cause][k][0] += bool(v)
                seen[cause][k][1] += 1
        self.line = copy.copy(cm)
        self.line.evidence = {}
        for c, e in cm.evidence.items():
            e2 = {**e, "p": dict(e["p"])}
            cls = self._type_of(c)
            n = self.n_type.get(cls, 0)
            if n:
                e2["prior"] = (K_PRIOR * e["prior"] + n_cause[c]) / (K_PRIOR + n)
            for k, (t, m) in seen[c].items():
                if k in e2["p"]:
                    e2["p"][k] = (K_CLUE * e["p"][k] + t) / (K_CLUE + m)
            self.line.evidence[c] = e2

    def _type_of(self, cause: str) -> str:
        return next(t for t in self.table.kb.defects if cause in self.table.causes(t))

    def confirmed(self, cls: str) -> int:
        return self.n_type.get(cls, 0)

    def posterior(self, cls: str, feats: dict) -> dict[str, float]:
        return self.line.posterior(cls, feats)

    def adjust(self, cls: str, feats: dict, p: dict[str, float]) -> dict[str, float]:
        """Llama's probabilities, shifted by how this line differs from the rule table."""
        if not self.n_type.get(cls):
            return dict(p)
        a, b = self.table.posterior(cls, feats), self.line.posterior(cls, feats)
        w = {c: p[c] * (b[c] / a[c] if a.get(c) else 1.0) for c in p}
        z = sum(w.values()) or 1.0
        return {c: v / z for c, v in w.items()}

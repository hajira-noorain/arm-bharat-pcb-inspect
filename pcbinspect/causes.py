"""The cause model: how likely each clue is under each cause (kb/evidence.yaml), combined by Bayes' rule.

It has three jobs: the plain lookup baseline Llama is measured against, the ground truth of the training-data
generator (llm/data/gen_cause.py), and the explanation shown in "details" (which clues point where, and where each
rule comes from). Clues are treated as independent given the cause (naive Bayes); unknown clues are left out.
"""
from __future__ import annotations

from pathlib import Path

import yaml

FEATURES = ("same_spot", "near_edge", "long_thin", "round", "many", "rising", "lane", "central", "roll_pitch")
DEFAULT_EVIDENCE = Path(__file__).resolve().parent.parent / "kb" / "evidence.yaml"
SAME_SPOT_MIN_BOARDS = 2
SAME_SPOT_MIN_SHARE = 0.25
MANY = 3


def features_from(ev: dict) -> dict[str, bool]:
    """Binary clues from the station's evidence (pcbinspect/evidence.py). A clue the station could not compute is
    absent, never False."""
    f: dict[str, bool] = {}
    rep = ev.get("repeat_same_spot")
    if isinstance(rep, dict) and rep.get("of"):
        f["same_spot"] = rep["boards"] >= SAME_SPOT_MIN_BOARDS and rep["boards"] / rep["of"] >= SAME_SPOT_MIN_SHARE
    if isinstance(ev.get("near_edge"), bool):
        f["near_edge"] = ev["near_edge"]
    if ev.get("shape") in ("long_thin", "round", "irregular"):
        f["long_thin"] = ev["shape"] == "long_thin"
        f["round"] = ev["shape"] == "round"
    if isinstance(ev.get("count_on_board"), int):
        f["many"] = ev["count_on_board"] >= MANY
    if ev.get("trend") in ("rising", "steady"):
        f["rising"] = ev["trend"] == "rising"
    line = ev.get("repeat_in_line")
    if isinstance(line, dict) and line.get("of"):
        f["lane"] = line["boards"] >= SAME_SPOT_MIN_BOARDS and line["boards"] / line["of"] >= SAME_SPOT_MIN_SHARE
    if isinstance(ev.get("central"), bool):
        f["central"] = ev["central"]
    if ev.get("roll_pitch") is True:     # only a found repeat is a clue: with small boards its absence says little
        f["roll_pitch"] = True
    return f


class CauseModel:
    def __init__(self, kb, path: Path = DEFAULT_EVIDENCE):
        self.kb = kb
        self.evidence: dict[str, dict] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["causes"]

    def causes(self, cls: str) -> list[str]:
        return [c["id"] for c in self.kb.entry(cls)["causes"]]

    def likelihood(self, cause: str, feats: dict[str, bool]) -> float:
        ev = self.evidence[cause]
        p = ev["p"]
        lk = ev["prior"]
        for k, v in feats.items():
            if k in ("long_thin", "round"):
                continue
            if k in p:
                lk *= p[k] if v else 1.0 - p[k]
        # shape is one clue with three values: long and thin, else round, else irregular
        if feats.get("long_thin") is True:
            lk *= p["long_thin"]
        elif "long_thin" in feats or "round" in feats:
            lk *= (1.0 - p["long_thin"]) * (p["round"] if feats.get("round") else 1.0 - p["round"])
        return lk

    def posterior(self, cls: str, feats: dict[str, bool]) -> dict[str, float]:
        lk = {c: self.likelihood(c, feats) for c in self.causes(cls)}
        z = sum(lk.values())
        return {c: v / z for c, v in lk.items()}

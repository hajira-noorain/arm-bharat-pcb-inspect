"""Training and test data for Llama's cause choice.

Generator 1 (training, and its held-out test split): pick a defect type, draw the true cause from the cause
model's prior (kb/evidence.yaml), then draw the clues that cause leaves, with some unknown. The label is the exact
posterior over that defect's causes given the clues shown, so ambiguity is taught as ambiguity. Whole clue
combinations are held out: a combination is either in training or in the test split, never both.

Generator 2 (test only, `alt_examples`): built differently on purpose. A hidden fault strength makes the clues
move together (the cause model assumes they do not), priors are flattened and some clues are flipped. A model that
only learned generator 1's tables loses accuracy here; one that learned the clue-to-cause directions should not.

    python -m llm.data.gen_cause --n 6000 --out <dir>     # writes train.jsonl, dev.jsonl, test.jsonl, alt.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from pathlib import Path

from pcbinspect.cause_prompt import build_evidence_prompt, cause_question
from pcbinspect.causes import CauseModel, features_from
from pcbinspect.kb import default_kb

ZONES = ["top-left", "top-centre", "top-right", "middle-left", "centre", "middle-right", "bottom-left",
         "bottom-centre", "bottom-right"]
SEVERITIES = ["critical", "major", "minor"]
UNKNOWN_P = {"same_spot": 0.15, "near_edge": 0.05, "trend": 0.40}
HOLDOUT_MOD = 8          # clue combinations are bucketed by hash: bucket 0 is test-only, bucket 1 dev-only


class Generator:
    def __init__(self, kb=None, alt: bool = False):
        self.kb = kb or default_kb()
        self.cm = CauseModel(self.kb)
        self.alt = alt
        self.types = list(self.kb.defects)

    def _p(self, cause: str, strength: float) -> dict:
        p = dict(self.cm.evidence[cause]["p"])
        if self.alt:      # clues move together with a hidden fault strength
            p = {k: min(0.97, max(0.03, v ** (0.4 + strength))) for k, v in p.items()}
        return p

    def _prior(self, cls: str) -> dict:
        pr = {c: self.cm.evidence[c]["prior"] for c in self.cm.causes(cls)}
        if self.alt:
            pr = {c: 0.5 * v + 0.5 / len(pr) for c, v in pr.items()}
        return pr

    def _clues(self, cause: str, rng: random.Random) -> dict:
        p = self._p(cause, rng.betavariate(2, 2))
        flip = (lambda b: (not b) if rng.random() < 0.08 else b) if self.alt else (lambda b: b)
        ev: dict = {}
        if rng.random() < UNKNOWN_P["same_spot"]:
            ev["repeat_same_spot"] = "unknown"
        else:
            of = rng.randint(6, 20)
            if flip(rng.random() < p["same_spot"]):
                ev["repeat_same_spot"] = {"boards": rng.randint(max(2, math.ceil(0.25 * of)), of), "of": of}
            else:
                ev["repeat_same_spot"] = {"boards": rng.randint(0, 1), "of": of}
        if ev["repeat_same_spot"] == "unknown":
            ev["repeat_in_line"] = "unknown"
        else:
            of = ev["repeat_same_spot"]["of"]
            ev["repeat_in_line"] = ({"boards": rng.randint(max(2, math.ceil(0.25 * of)), of), "of": of}
                                    if flip(rng.random() < p["lane"]) else {"boards": rng.randint(0, 1), "of": of})
        ev["central"] = "unknown" if rng.random() < UNKNOWN_P["near_edge"] else flip(rng.random() < p["central"])
        ev["roll_pitch"] = flip(rng.random() < p["roll_pitch"])
        ev["near_edge"] = "unknown" if rng.random() < UNKNOWN_P["near_edge"] else flip(rng.random() < p["near_edge"])
        if flip(rng.random() < p["long_thin"]):
            ev["shape"] = "long_thin"
        else:
            ev["shape"] = "round" if flip(rng.random() < p["round"]) else "irregular"
        ev["count_on_board"] = rng.randint(3, 8) if flip(rng.random() < p["many"]) else rng.randint(1, 2)
        ev["trend"] = ("unknown" if rng.random() < UNKNOWN_P["trend"] else
                       "rising" if flip(rng.random() < p["rising"]) else "steady")
        return ev

    def _defect(self, cls: str, rng: random.Random, did: str) -> tuple[dict, str]:
        pr = self._prior(cls)
        cause = rng.choices(list(pr), weights=list(pr.values()))[0]
        d = {"id": did, "type": cls, "severity": rng.choice(SEVERITIES), "size_mm": round(rng.uniform(0.1, 2.0), 2),
             "location": rng.choice(ZONES), "confidence": round(rng.uniform(0.5, 0.99), 2),
             "evidence": self._clues(cause, rng)}
        return d, cause

    def example(self, rng: random.Random, i: int) -> dict:
        cls = rng.choice(self.types)
        n = rng.choice([1, 1, 2, 3])
        pos = rng.randrange(n)
        defects, true_cause = [], None
        others = [t for t in self.types if t != cls]
        for k in range(n):
            t = cls if k == pos else rng.choice(others)
            d, c = self._defect(t, rng, f"D{k + 1}")
            defects.append(d)
            if k == pos:
                true_cause = c
        types = [d["type"] for d in defects]
        for d in defects:
            d["evidence"]["co_occurring"] = sorted(set(types) - {d["type"]})
        target = defects[pos]
        feats = features_from(target["evidence"])
        combo = cls + "|" + ",".join(f"{k}={int(v)}" for k, v in sorted(feats.items()))
        return {"id": f"{'alt' if self.alt else 'g1'}-{i}", "generator": "alt" if self.alt else "g1",
                "record": {"board_id": f"B{10000 + i}", "verdict": rng.choice(["FAIL", "REVIEW"]), "defects": defects},
                "defect_id": target["id"], "true_cause": true_cause, "combo": combo,
                "posterior": self.cm.posterior(cls, feats), "variant": rng.randrange(4)}


def held_out(combo: str) -> bool:
    return split_of(combo) == "test"


def split_of(combo: str) -> str:
    """Which split a clue combination belongs to. Dev has its own unseen combinations, so choosing the best
    checkpoint measures generalisation, not fit on combinations already trained on."""
    b = zlib.crc32(combo.encode()) % HOLDOUT_MOD
    return "test" if b == 0 else "dev" if b == 1 else "train"


def examples(n: int, seed: int = 0, split: str = "train", kb=None) -> list[dict]:
    assert split in ("train", "dev", "test"), split
    g, rng, out, i = Generator(kb), random.Random(seed), [], 0
    while len(out) < n:
        ex = g.example(rng, i)
        i += 1
        if split_of(ex["combo"]) == split:
            out.append(ex)
    return out


def alt_examples(n: int, seed: int = 0, kb=None) -> list[dict]:
    g, rng = Generator(kb, alt=True), random.Random(seed)
    return [g.example(rng, i) for i in range(n)]


def training_rows(ex: dict, kb) -> list[dict]:
    """One yes/no question per candidate cause; the target is the probability of "yes"."""
    prompt = build_evidence_prompt(ex["record"], kb, variant=ex.get("variant", 0))
    return [{"id": ex["id"], "cause": c, "prompt": prompt + cause_question(ex["defect_id"], c, kb), "p_yes": p}
            for c, p in ex["posterior"].items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="training examples (dev and test get n/10 each)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kb = default_kb()
    sets = {"train": examples(a.n, 11, "train", kb), "dev": examples(a.n // 10, 12, "dev", kb),
            "test": examples(a.n // 10, 13, "test", kb), "alt": alt_examples(a.n // 10, 14, kb)}
    train_combos = {e["combo"] for e in sets["train"]}
    assert not (train_combos & {e["combo"] for e in sets["test"]})
    assert not (train_combos & {e["combo"] for e in sets["dev"]})
    for name, exs in sets.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e) + "\n")
        print(name, len(exs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

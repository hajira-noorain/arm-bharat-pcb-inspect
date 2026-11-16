"""Training data for the "why" line (pcbinspect/why.py): facts from a generated defect, the cause the station would
show (the most probable one), the clue code says favours it, and a correct fill-in sentence naming that clue.
Drawn from the training split of the cause generator, so no test clue combination appears here."""
from __future__ import annotations

import random

from llm.data.gen_cause import examples as cause_examples
from pcbinspect.causes import CauseModel
from pcbinspect.kb import default_kb
from pcbinspect.why import why_facts, why_prompt

PHRASES = {
    "same_spot_boards": ["The defect sits at the same place on {same_spot_boards} of the last {same_spot_of} boards, which points to {cause}.",
                         "{cause} is likely: the same spot was hit on {same_spot_boards} of the last {same_spot_of} boards.",
                         "Seen at this exact spot on {same_spot_boards} of the last {same_spot_of} boards, so {cause} is the likely cause."],
    "lane_boards": ["The same type lines up on {lane_boards} of the last {lane_of} boards, which points to {cause}.",
                    "{cause} is likely: it falls in a line with the same type on {lane_boards} of the last {lane_of} boards."],
    "edge": ["It is {edge}, which points to {cause}.", "{cause} is likely because the defect is {edge}."],
    "shape": ["Its shape is {shape}, which fits {cause}.", "{cause} is likely: the defect is {shape}."],
    "count": ["There are {count} of this type on the board, which points to {cause}.",
              "{cause} is likely: {count} of them on this one board."],
    "trend": ["This type is {trend} on recent boards, which points to {cause}.",
              "{cause} is likely because this type is {trend} on recent boards."],
    "middle": ["It is {middle}, which points to {cause}.", "{cause} is likely: the defect is {middle}."],
    "roll": ["Another one {roll}, which points to {cause}.", "{cause} is likely: another one {roll}."],
}


def examples(n: int, seed: int = 0, split: str = "train") -> list[dict]:
    kb = default_kb()
    cm = CauseModel(kb)
    rng = random.Random(seed)
    out = []
    for ex in cause_examples(n * 3, seed + 1000, split, kb):
        d = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])
        cause = max(ex["posterior"], key=ex["posterior"].get)
        f, key = why_facts(d["type"], d["evidence"], cause, cm, kb)
        if key is None:
            continue
        out.append({"facts": f, "key": key, "target": rng.choice(PHRASES[key])})
        if len(out) == n:
            break
    return out


def training_rows(ex: dict) -> dict:
    return {"prompt": why_prompt(ex["facts"]), "completion": ex["target"]}

"""The "why" line under each cause: one sentence, checkable by the station, naming the clue that points to it.

Which clue to name is decided by code: the clue whose observed value most favours the chosen cause over the other
causes of that defect type, by the rule table's likelihoods (pcbinspect/causes.py). Llama only words it, as a
fill-in sentence ({same_spot_boards}, {cause}, ...) like the shift answers (pcbinspect/sentences.py); the checker
requires the cause and that clue, and refuses typed numbers and unknown fields. A refused sentence is replaced by
the standard one, so the operator always sees the right clue.
"""
from __future__ import annotations

import math

from .causes import features_from
from .sentences import DIGIT, PLACEHOLDER, SENTENCE_END, render

SYSTEM = ("You tell a factory operator, in one sentence, which clue points to the likely cause of a defect. "
          "Write every number or word from the facts as {name}, using the names listed; never type a number.")
KEY_OF = {"same_spot": "same_spot_boards", "lane": "lane_boards", "near_edge": "edge", "long_thin": "shape",
          "round": "shape", "many": "count", "rising": "trend", "central": "middle", "roll_pitch": "roll"}


def _factor(p: dict, k: str, v: bool, feats: dict) -> float:
    if k == "long_thin":
        return p["long_thin"] if v else 1.0 - p["long_thin"]
    if k == "round":
        return 1.0 if feats.get("long_thin") else (p["round"] if v else 1.0 - p["round"])
    return p[k] if v else 1.0 - p[k]


def why_facts(cls: str, ev: dict, cause_id: str, cm, kb) -> tuple[dict, str | None]:
    """The facts Llama may use, and the field it must use (None: no clue favours this cause)."""
    ev = ev or {}
    f: dict = {"defect": cls.replace("_", " "), "cause": kb.by_id[cause_id]["short"]}
    rep, line = ev.get("repeat_same_spot"), ev.get("repeat_in_line")
    if isinstance(rep, dict):
        f["same_spot_boards"], f["same_spot_of"] = rep["boards"], rep["of"]
    if isinstance(line, dict):
        f["lane_boards"], f["lane_of"] = line["boards"], line["of"]
    if isinstance(ev.get("near_edge"), bool):
        f["edge"] = "near the board edge" if ev["near_edge"] else "away from the board edge"
    if ev.get("shape") in ("long_thin", "round", "irregular"):
        f["shape"] = {"long_thin": "long and thin", "round": "round", "irregular": "irregular"}[ev["shape"]]
    if isinstance(ev.get("count_on_board"), int):
        f["count"] = ev["count_on_board"]
    if ev.get("trend") in ("rising", "steady"):
        f["trend"] = ev["trend"]
    if isinstance(ev.get("central"), bool):
        f["middle"] = "in the middle of the board" if ev["central"] else "away from the middle of the board"
    if ev.get("roll_pitch") is True:
        f["roll"] = "repeats a laminator roll's length away"
    feats = features_from(ev)
    others = [c for c in cm.causes(cls) if c != cause_id]
    best, key = 0.0, None
    for k, v in feats.items():
        mine = _factor(cm.evidence[cause_id]["p"], k, v, feats)
        theirs = sum(_factor(cm.evidence[c]["p"], k, v, feats) for c in others) / max(len(others), 1)
        score = math.log(max(mine, 1e-9) / max(theirs, 1e-9))
        if score > best + 1e-9 and KEY_OF[k] in f:
            best, key = score, KEY_OF[k]
    return f, key


def why_prompt(f: dict, key: str | None = None) -> str:
    """With key, the prompt also says which clue to name (from v3.1 on); without, the prompt v3 was trained on."""
    listed = "; ".join(f"{k} = {v}" for k, v in f.items())
    name = f"\nName this clue: {key}" if key else ""
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + f"Facts: {listed}{name}" +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def check_why(tpl: str, f: dict, key: str | None) -> list[str]:
    t = (tpl or "").strip()
    used = [m.group(1) for m in PLACEHOLDER.finditer(t)]
    if not t or not used:
        return ["the sentence gives no fact" if t else "empty"]
    out = []
    if DIGIT.search(PLACEHOLDER.sub("", t)):
        out.append("a number was typed instead of written as a field")
    unknown = sorted(set(used) - set(f))
    if unknown:
        out.append(f"fields that do not exist: {unknown}")
    need = {"cause"} | ({key} if key else set())
    if key == "same_spot_boards":
        need.add("same_spot_of")
    if key == "lane_boards":
        need.add("lane_of")
    missing = sorted(need - set(used))
    if missing:
        out.append(f"does not name the clue: {missing}")
    if len(SENTENCE_END.findall(t + " ")) > 2:
        out.append("more than two sentences")
    return out


STANDARD = {
    "same_spot_boards": "{cause}: the same spot was hit on {same_spot_boards} of the last {same_spot_of} boards.",
    "lane_boards": "{cause}: the same type lines up on {lane_boards} of the last {lane_of} boards.",
    "edge": "{cause}: the defect is {edge}.",
    "shape": "{cause}: its shape is {shape}.",
    "count": "{cause}: there are {count} of this type on the board.",
    "trend": "{cause}: this type is {trend} on recent boards.",
    "middle": "{cause}: the defect is {middle}.",
    "roll": "{cause}: another one {roll}.",
}


def standard_why(f: dict, key: str | None) -> str:
    return render(STANDARD[key], f) if key else ""

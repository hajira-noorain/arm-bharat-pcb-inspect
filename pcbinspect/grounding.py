"""Deterministic grounding checker for Llama advice.

Llama's job is narrow: for each defect, choose the most likely cause and the next action from the
knowledge-base entries it was given, order them, and write a one-to-two sentence summary.
The operator sees KB text verbatim for every cause/action, so instructions can never be invented.
This checker enforces that contract and rejects anything else:

  1. output must parse as JSON with the agreed schema
  2. every referenced defect id must exist in the record
  3. every cause/action id must be one the KB allows for THAT defect's type
  4. every number in the summary must appear in the record (sizes, confidences, counts)
  5. the summary may not name a defect type that is not on the board
  6. every critical defect must be covered (otherwise it is filled from the template and flagged)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .kb import KnowledgeBase

NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def number_filter(record: dict, extra_numbers: list[float] | None = None):
    """allow(text_so_far, piece) for LogitLlama.write(): a token is vetoed if it would complete, or start,
    a number that check() would reject (not in the record or the history given to the model). The checker
    still runs afterwards; this only stops the model from writing what the checker would throw away."""
    return number_filter_values(_record_numbers(record) | {round(float(x), 2) for x in (extra_numbers or [])},
                                integers_may_round=True)


def number_filter_values(allowed: set[float], integers_may_round: bool = False):
    """The same decoding filter for any set of permitted numbers (e.g. the numbers in database rows)."""
    allowed = {round(float(x), 2) for x in allowed}
    ints = {round(x) for x in allowed} if integers_may_round else {x for x in allowed if float(x).is_integer()}
    forms = {f"{x:g}" for x in allowed} | {f"{x:.2f}" for x in allowed} | {f"{x:.1f}" for x in allowed} | \
        {str(int(i)) for i in ints}

    def ok_value(tok: str) -> bool:
        v = round(float(tok), 2)
        return v in allowed or (v.is_integer() and v in ints)

    def allow(text: str, piece: str) -> bool:
        if not any(ch.isdigit() for ch in piece) and not (text[-1:].isdigit() or text[-1:] == "."):
            return True
        new = text + piece
        tail = re.search(r"(?<![\w.])(\d+(?:\.\d*)?)$", new)
        body = new[:tail.start()] if tail else new
        if any(not ok_value(m) for m in NUM_RE.findall(body)):
            return False
        return True if tail is None else any(f.startswith(tail.group(1)) for f in forms)
    return allow


@dataclass
class GroundingResult:
    ok: bool
    advice: dict | None
    reasons: list[str] = field(default_factory=list)
    filled_from_template: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict | None:
    """Return the first balanced top-level JSON object in text, or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _record_numbers(record: dict) -> set[float]:
    nums: set[float] = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            nums.add(round(float(v), 2))
            # allow percentages of confidences (0.87 -> 87)
            if 0 <= v <= 1:
                nums.add(round(float(v) * 100, 0))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            for m in NUM_RE.findall(v):
                nums.add(round(float(m), 2))

    walk(record)
    nums.add(float(len(record.get("defects", []))))
    return nums


KB_ID_RE = re.compile(r"\b[A-Z]{3,6}-[CA]\d+\b")


def _normalise_id(value, allowed: set[str], ordered: list[str] | None = None):
    """Map the model's choice to a KB ID, or return it unchanged so it is rejected.

    - an option NUMBER (numbered prompt): 1-based index into this defect's own option list;
      out of range -> unchanged -> rejected
    - an echoed option line ('causes: SHORT-C1: under-etching'): formatting noise, accepted only if
      the field contains exactly one KB ID and that ID is allowed for this defect
    """
    if ordered is not None and not isinstance(value, bool):
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            k = int(value)
            return ordered[k - 1] if 1 <= k <= len(ordered) else f"option {k} (out of range)"
    if not isinstance(value, str) or value in allowed:
        return value
    ids = set(KB_ID_RE.findall(value))
    if len(ids) == 1 and next(iter(ids)) in allowed:
        return next(iter(ids))
    return value


def check(raw_text: str, record: dict, kb: KnowledgeBase,
          extra_numbers: list[float] | None = None) -> GroundingResult:
    """extra_numbers: other facts given to the model in the prompt (e.g. history counts)."""
    reasons: list[str] = []
    obj = extract_json(raw_text)
    if obj is None:
        return GroundingResult(False, None, ["output is not a JSON object"])

    summary = obj.get("summary")
    steps = obj.get("steps")
    if not isinstance(summary, str) or not summary.strip():
        reasons.append("missing summary")
    if not isinstance(steps, list) or not steps:
        reasons.append("missing steps")
    if reasons:
        return GroundingResult(False, None, reasons)

    defects = {d["id"]: d for d in record["defects"]}
    clean_steps = []
    seen: set[str] = set()
    for s in steps:
        if not isinstance(s, dict):
            reasons.append("step is not an object")
            continue
        did, cause, action = s.get("defect"), s.get("cause"), s.get("action")
        if did not in defects:
            reasons.append(f"step cites unknown defect {did!r}")
            continue
        allowed_c, allowed_a = kb.allowed_ids(defects[did]["type"])
        entry = kb.entry(defects[did]["type"])
        cause = _normalise_id(cause, allowed_c, [c["id"] for c in entry["causes"]])
        action = _normalise_id(action, allowed_a, [a["id"] for a in entry["actions"]])
        if cause not in allowed_c:
            reasons.append(f"{did}: cause {cause!r} not allowed for {defects[did]['type']}")
            continue
        if action not in allowed_a:
            reasons.append(f"{did}: action {action!r} not allowed for {defects[did]['type']}")
            continue
        if did in seen:
            continue  # duplicate step for same defect: keep the first, not an error
        seen.add(did)
        clean_steps.append({"defect": did, "cause": cause, "action": action})

    # numbers in the summary must come from the record
    allowed_nums = _record_numbers(record) | {round(float(x), 2) for x in (extra_numbers or [])}
    for m in NUM_RE.findall(summary):
        v = round(float(m), 2)
        if v not in allowed_nums and not (v.is_integer() and v in {round(x) for x in allowed_nums}):
            reasons.append(f"summary contains number {m} that is not in the record")

    # the summary must not mention defect types absent from the board
    present = {d["type"] for d in record["defects"]}
    low = summary.lower()
    for cls, words in kb.all_class_names().items():
        if cls in present:
            continue
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                # "open"/"short" are also ordinary English words; only flag them in a defect context
                if cls in ("open", "short") and not re.search(
                        rf"\b{cls} (circuit|trace|conductor)s?\b", low):
                    continue
                reasons.append(f"summary mentions '{w}' but no {cls} was found")
                break

    if reasons:
        return GroundingResult(False, None, reasons)

    # completeness: every critical defect must have a step; fill gaps from the template
    filled = []
    for did, d in defects.items():
        if did not in seen and d.get("severity") == "critical":
            c_ids, a_ids = kb.entry(d["type"])["causes"], kb.entry(d["type"])["actions"]
            clean_steps.append({"defect": did, "cause": c_ids[0]["id"], "action": a_ids[0]["id"]})
            filled.append(did)

    return GroundingResult(True, {"summary": summary.strip(), "steps": clean_steps}, [], filled)

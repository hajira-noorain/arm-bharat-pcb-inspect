"""Customer acceptance criteria -> verified, executable inspection rules.

A shop receives criteria in plain language ("Nicks reducing conductor width by more than 20% are
rejectable", "No shorts permitted", "Inspect to Class 3"). Llama (on ExecuTorch, on the station)
fills the fields of a small rule language; everything after that is deterministic:

  1. schema      every field is from the closed vocabulary, the value has the right kind
  2. grounding   the rule's number must be a number written in the clause (after unit conversion:
                 mm / mil / micron / percent), so 0.1 mm can't silently become 0.01 mm
  3. consistency the defect word, comparison words and action words in the clause agree with the
                 rule (keyword evidence; a disagreement sends the rule to the operator, it is not
                 silently "fixed")
  4. capability  the rule is checked against the station's MEASURED detection limit (pod.py): a
                 rule that needs the station to find defects smaller than it is proven to find is
                 marked NOT GUARANTEED, with the size gap stated
  5. execution   (station) the rule is replayed on logged inspections: "would have changed N boards"

The rule language
  kind     limit | class
  defect   one of BARE_CLASSES, or "any"
  measure  always | size_mm | width_loss_pct | spacing_loss_pct | count
  op       ">" | ">="   (the rule fires when measure op value; not used for "always")
  value    mm for size_mm, percent for *_pct, defects per board for count
  action   reject | rework | review | accept
  customer_class  1 | 2 | 3   (kind == class only)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .record import BARE_CLASSES

MEASURES = ("always", "size_mm", "width_loss_pct", "spacing_loss_pct", "count")
OPS = (">", ">=")
ACTIONS = ("reject", "rework", "review", "accept")
ACTION_RANK = {"accept": 0, "review": 1, "rework": 2, "reject": 3}
MM_PER_MIL = 0.0254

DEFECT_WORDS: dict[str, tuple[str, ...]] = {
    "open": ("open circuit", "open", "broken trace", "broken conductor", "break in", "severed", "cut trace"),
    "short": ("short circuit", "short", "bridging", "bridge", "bridged"),
    "mouse_bite": ("mouse bite", "mousebite", "nick", "notch", "edge bite"),
    "spur": ("spur", "protrusion", "projection"),
    "spurious_copper": ("spurious copper", "copper splash", "stray copper", "extraneous copper", "copper residue",
                        "copper island", "residual copper"),
    "pin_hole": ("pin hole", "pinhole", "pin-hole"),
    "missing_hole": ("missing hole", "undrilled", "missing drill", "unplated hole missing"),
    "missing_pad": ("missing pad", "damaged pad", "pad missing", "etched-away pad", "eroded pad"),
    "hole_breakout": ("hole breakout", "breakout", "broken annular ring", "annular ring break", "tangency"),
    "scratch": ("scratch", "scratched conductor", "scuff"),
    "foreign_object": ("foreign object", "foreign material", "debris", "fibre", "fiber", "particle"),
}
DISPLAY = {"open": "open", "short": "short", "mouse_bite": "mouse bite", "spur": "spur",
           "spurious_copper": "spurious copper", "pin_hole": "pin hole", "missing_hole": "missing hole",
           "missing_pad": "missing pad", "hole_breakout": "hole breakout", "scratch": "scratch",
           "foreign_object": "foreign object", "any": "any defect"}
MEASURE_WORDS = {
    "width_loss_pct": ("width", "conductor width", "trace width", "track width", "line width"),
    "spacing_loss_pct": ("spacing", "clearance", "gap", "isolation"),
    "count": ("per board", "per panel", "per pcb", "number of", "count", "more than one", "more than two"),
}
GE_WORDS = ("at least", "or more", "or larger", "or greater", "or above", "and above", "and larger",
            "not less than", "minimum of", ">=", "≥", "or bigger", "and over", "or over")
GT_WORDS = ("more than", "greater than", "larger than", "bigger than", "exceed", "exceeding", "exceeds",
            "over", "above", "beyond", "up to", "no more than", "not exceed", "maximum", "max", "at most", ">")
ACTION_WORDS = {
    "reject": ("reject", "rejectable", "scrap", "not acceptable", "unacceptable", "not permitted",
               "not allowed", "shall not", "must not", "no ", "fail", "nonconforming", "non-conforming",
               "is a defect", "are defects", "not accepted"),
    "rework": ("rework", "repair", "touch up", "touch-up", "correct", "fix"),
    "review": ("review", "flag", "hold", "manual inspection", "inspect manually", "engineer", "check by",
               "quarantine", "mrb"),
    "accept": ("acceptable", "accept", "allowed", "permitted", "permissible", "ok", "pass"),
}
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*(mm|millimet(?:er|re)s?|mils?|thou|µm|um|"
                    r"microns?|micromet(?:er|re)s?|%|percent|per\s*cent)?", re.I)
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "zero": 0, "none": 0}


@dataclass
class Rule:
    kind: str = "limit"
    defect: str = "any"
    measure: str = "always"
    op: str | None = None
    value: float | None = None
    action: str = "reject"
    customer_class: int | None = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Rule":
        return Rule(**{k: d.get(k) for k in Rule.__dataclass_fields__ if k in d})


def rule_from_short(d: dict) -> Rule:
    """Benchmark shorthand {k, d, m, op, v, a, c} -> Rule."""
    if d.get("k") == "class":
        return Rule(kind="class", customer_class=int(d["c"]), action="reject")
    return Rule(defect=d["d"], measure=d["m"], op=d.get("op"),
                value=None if d.get("v") is None else float(d["v"]), action=d["a"])


def same_rule(a: Rule | None, b: Rule | None) -> bool:
    if a is None or b is None:
        return False
    if a.kind != b.kind:
        return False
    if a.kind == "class":
        return a.customer_class == b.customer_class
    if (a.defect, a.measure, a.action) != (b.defect, b.measure, b.action):
        return False
    if a.measure == "always":
        return True
    return a.op == b.op and a.value is not None and b.value is not None and \
        abs(a.value - b.value) <= 1e-6 + 1e-3 * abs(b.value)


def schema_errors(r: Rule) -> list[str]:
    e = []
    if r.kind == "class":
        if r.customer_class not in (1, 2, 3):
            e.append("customer class must be 1, 2 or 3")
        return e
    if r.kind != "limit":
        return [f"unknown kind {r.kind!r}"]
    if r.defect not in BARE_CLASSES and r.defect != "any":
        e.append(f"unknown defect {r.defect!r}")
    if r.measure not in MEASURES:
        e.append(f"unknown measure {r.measure!r}")
    if r.action not in ACTIONS:
        e.append(f"unknown action {r.action!r}")
    if r.measure != "always":
        if r.op not in OPS:
            e.append(f"comparison must be one of {OPS}")
        if r.value is None or not isinstance(r.value, (int, float)) or r.value < 0:
            e.append("value must be a non-negative number")
        elif r.measure.endswith("_pct") and r.value > 100:
            e.append("percentage above 100")
        elif r.measure == "size_mm" and r.value > 50:
            e.append("size above 50 mm is not a PCB defect size")
    return e


def render(r: Rule) -> str:
    """Deterministic English for the operator to confirm (never generated by the model)."""
    if r.kind == "class":
        return f"Judge boards to IPC-style Class {r.customer_class}."
    what = ("any defect" if r.defect == "any" else "spurious copper" if r.defect == "spurious_copper"
            else f"a {DISPLAY.get(r.defect, r.defect)}")
    verb = {"reject": "Reject the board for", "rework": "Send for rework", "review": "Send for manual review",
            "accept": "Accept"}[r.action]
    if r.action == "rework":
        verb = "Send the board for rework for"
    if r.action == "review":
        verb = "Send the board for manual review for"
    if r.measure == "always":
        return f"{verb} {what}, whatever its size." if r.action != "accept" else f"Accept {what} of any size."
    cmp_ = "more than" if r.op == ">" else "at least"
    v = r.value
    tail = {"size_mm": f"when it is {cmp_} {v:g} mm in size",
            "width_loss_pct": f"when it reduces conductor width by {cmp_} {v:g}%",
            "spacing_loss_pct": f"when it reduces conductor spacing by {cmp_} {v:g}%",
            "count": f"when there are {cmp_} {v:g} on one board"}[r.measure]
    return f"{verb} {what} {tail}."


# ---------------------------------------------------------------- execution

def fires(r: Rule, d: dict, count_on_board: int = 1) -> bool:
    """d: {"cls", "size_mm", "width_loss_pct", "spacing_loss_pct"} (missing measurement -> None)."""
    if r.kind != "limit" or (r.defect != "any" and d.get("cls") != r.defect):
        return False
    if r.measure == "always":
        return True
    x = count_on_board if r.measure == "count" else d.get(r.measure)
    if x is None:
        return r.action != "accept"  # unmeasured: a reject/rework/review rule cannot be cleared
    return x > r.value if r.op == ">" else x >= r.value


def apply_rules(rules: list[Rule], defects: list[dict]) -> tuple[list[str | None], str]:
    """Per-defect action (worst firing rule; an explicit 'accept' rule clears only if nothing else
    fires) and the board action."""
    counts: dict[str, int] = {}
    for d in defects:
        counts[d.get("cls")] = counts.get(d.get("cls"), 0) + 1
    out = []
    for d in defects:
        acts = [r.action for r in rules if fires(r, d, counts.get(d.get("cls"), 0))]
        worst = max((a for a in acts if a != "accept"), key=ACTION_RANK.get, default=None)
        out.append(worst or ("accept" if acts else None))
    board = max((a for a in out if a), key=ACTION_RANK.get, default="accept")
    return out, board


# ---------------------------------------------------------------- evidence from the clause text

def numbers_in(text: str) -> list[tuple[float, str | None]]:
    t = text.lower()
    out = []
    for m in NUM_RE.finditer(t):
        unit = (m.group(2) or "").lower().replace(" ", "")
        if unit.startswith("mil") and not unit.startswith("milli") or unit == "thou":
            unit = "mil"
        elif unit.startswith(("µm", "um", "micron", "micromet")):
            unit = "micron"
        elif unit.startswith(("mm", "millimet")):
            unit = "mm"
        elif unit in ("%", "percent", "percent", "per cent", "percent"):
            unit = "%"
        elif unit.startswith("per"):
            unit = "%"
        else:
            unit = None
        out.append((float(m.group(1)), unit))
    for w, v in WORD_NUM.items():
        if re.search(rf"\b{w}\b", t):
            out.append((float(v), None))
    return out


def to_canonical(number: float, unit: str | None, measure: str) -> float | None:
    if measure == "size_mm":
        v = {"mm": number, "mil": number * MM_PER_MIL, "micron": number / 1000.0}.get(unit or "mm")
        return None if v is None else round(v, 6)  # micron-level: no 0.0761999.. knife edges
    if measure.endswith("_pct"):
        return number if unit in ("%", None) else None
    if measure == "count":
        return number if unit is None else None
    return None


def _has(text: str, words) -> bool:
    """Whole-word match (plural -s/-es allowed) for words; plain substring for symbols/phrases ending in
    a non-letter."""
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?:s|es)?(?![a-z])", text) if w[-1].isalpha()
               else w in text for w in words)


def defects_mentioned(text: str) -> set[str]:
    t = text.lower()
    found = {c for c, ws in DEFECT_WORDS.items() if _has(t, ws)}
    if "short" in found and re.search(r"\bshort(?:est)?\s+(?:side|edge|length)", t):
        found.discard("short")
    return found


def verify(r: Rule, clause: str) -> dict:
    """Deterministic checks of a compiled rule against the clause it came from.

    status: invalid (schema), ungrounded (number not in clause), check (evidence disagrees: the
    operator must confirm), verified. Never edits the rule."""
    t = clause.lower()
    errs = schema_errors(r)
    if errs:
        return {"status": "invalid", "problems": errs}
    probs = []
    if r.kind == "class":
        m = re.search(r"class\s*([123])\b", t)
        if not m or int(m.group(1)) != r.customer_class:
            probs.append("class number not stated in the clause")
        return {"status": "check" if probs else "verified", "problems": probs}
    if r.measure != "always":
        cands = [to_canonical(n, u, r.measure) for n, u in numbers_in(t)]
        if not any(c is not None and abs(c - r.value) <= 1e-6 + 1e-3 * abs(r.value) for c in cands):
            return {"status": "ungrounded",
                    "problems": [f"value {r.value:g} ({r.measure}) does not match any number written in the clause"]}
    mentioned = defects_mentioned(t)
    if r.defect != "any" and r.defect not in mentioned:
        probs.append(f"clause does not mention {DISPLAY[r.defect]}")
    if r.defect == "any" and mentioned and not re.search(r"\b(any|all|every)\b", t):
        probs.append(f"clause names {', '.join(sorted(mentioned))} but rule applies to any defect")
    # (count rules need no word check: grounding already requires a unitless number)
    if r.measure in ("width_loss_pct", "spacing_loss_pct") and not _has(t, MEASURE_WORDS[r.measure]):
        probs.append(f"clause does not mention {r.measure.replace('_pct', '').replace('_', ' ')}")
    if r.measure == "always" and any(u is not None for _, u in numbers_in(t)):
        probs.append("clause gives a size or percentage but the rule ignores it")
    if r.measure != "always":
        ge, gt = _has(t, GE_WORDS), _has(t, GT_WORDS)
        if ge and not gt and r.op != ">=":
            probs.append("clause says 'at least / or more' but rule uses 'more than'")
        if gt and not ge and r.op != ">":
            probs.append("clause says 'more than / up to / exceed' but rule uses 'at least'")
    acts = {a for a, ws in ACTION_WORDS.items() if _has(t, ws)}
    if "reject" in acts and "accept" in acts:
        acts.discard("accept")  # "acceptable up to X" phrasing: the limit rejects beyond X
    if acts == {"accept"} and r.measure != "always":
        acts = set()  # "X is acceptable up to N": only the limit is stated; the action beyond it is implied
    if acts and r.action not in acts:
        probs.append(f"clause action words suggest {'/'.join(sorted(acts))}, rule says {r.action}")
    return {"status": "check" if probs else "verified", "problems": probs}


# ---------------------------------------------------------------- baseline: keywords only

def keyword_compile(clause: str) -> Rule | None:
    """Rule-of-thumb parser used as the baseline the LLM must beat."""
    t = clause.lower()
    m = re.search(r"class\s*([123])\b", t)
    if m and not defects_mentioned(t):
        return Rule(kind="class", customer_class=int(m.group(1)), action="reject")
    ds = defects_mentioned(t)
    defect = sorted(ds)[0] if len(ds) == 1 else "any"
    nums = [(n, u) for n, u in numbers_in(t)]
    measure, value = "always", None
    pct = [n for n, u in nums if u == "%"]
    size = [(n, u) for n, u in nums if u in ("mm", "mil", "micron")]
    if pct:
        measure = "spacing_loss_pct" if _has(t, MEASURE_WORDS["spacing_loss_pct"]) else "width_loss_pct"
        value = pct[0]
    elif size:
        measure, value = "size_mm", to_canonical(size[0][0], size[0][1], "size_mm")
    elif _has(t, MEASURE_WORDS["count"]) and nums:
        measure, value = "count", nums[0][0]
    op = None
    if measure != "always":
        op = ">=" if _has(t, GE_WORDS) and not _has(t, GT_WORDS) else ">"
    acts = [a for a in ("rework", "review", "reject", "accept") if _has(t, ACTION_WORDS[a])]
    action = acts[0] if acts else "reject"
    if action == "accept" and _has(t, ACTION_WORDS["reject"]):
        action = "reject"
    return Rule(defect=defect, measure=measure, op=op, value=value, action=action)


# ---------------------------------------------------------------- capability vs the measured limit

def capability(r: Rule, limits_mm: dict[str, float | None], min_trace_width_mm: float | None = None,
               min_spacing_mm: float | None = None) -> dict:
    """Can the station enforce this rule, given its measured detection limits (mm, per defect type)?

    guaranteed      every defect the rule targets is at or above the proven detection limit
    limited         the rule targets defects of any size; the station proves detection only from L up
    not_guaranteed  the rule targets sizes below the proven limit: the gap is stated
    unmeasured      no detection limit has been measured for this defect type on this rig
    """
    if r.kind != "limit" or r.action == "accept":
        return {"status": "guaranteed", "reason": "no detection needed"}
    types = list(limits_mm) if r.defect == "any" else [r.defect]
    lims = [limits_mm.get(t) for t in types]
    if any(v is None for v in lims) or not lims:
        missing = [t for t, v in zip(types, lims) if v is None]
        return {"status": "unmeasured",
                "reason": f"no measured detection limit for {', '.join(DISPLAY.get(t, t) for t in missing) or 'this type'}"}
    L = max(lims)
    if r.measure in ("always", "count"):
        return {"status": "limited", "limit_mm": L,
                "reason": f"proven to find these from {L:.2f} mm; smaller ones may be missed"}
    if r.measure == "size_mm":
        need = r.value
    elif r.measure == "width_loss_pct":
        if not min_trace_width_mm:
            return {"status": "unmeasured", "reason": "design has no minimum trace width"}
        need = r.value / 100.0 * min_trace_width_mm
    else:
        if not min_spacing_mm:
            return {"status": "unmeasured", "reason": "design has no minimum spacing"}
        need = r.value / 100.0 * min_spacing_mm
    if need >= L:
        return {"status": "guaranteed", "limit_mm": L, "needed_mm": need,
                "reason": f"rule acts from {need:.2f} mm; station proven from {L:.2f} mm"}
    return {"status": "not_guaranteed", "limit_mm": L, "needed_mm": need,
            "reason": f"rule needs defects from {need:.2f} mm found, but the station is only proven from "
                      f"{L:.2f} mm: those between {need:.2f} and {L:.2f} mm may be missed"}

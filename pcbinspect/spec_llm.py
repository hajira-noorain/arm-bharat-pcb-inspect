"""Llama (ExecuTorch, on the station) fills the rule fields from a plain-language acceptance clause.

Step-wise slot filling on one KV cache: the clause is prefilled once, then each field is a short
continuation ("Defect: ", "Measure: ", ...) that stops at the newline. Answers are WORDS from a
listed vocabulary (not option numbers, which showed a strong position-1 bias in our advice tests);
the lists are shuffled per clause in evaluation. The model only reads language; every number is
re-parsed from its answer, unit-converted and then checked against the clause by spec.verify().
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from .advisor import HDR
from .spec import DISPLAY, Rule, defects_mentioned, numbers_in, to_canonical

SYSTEM = ("You read one PCB acceptance requirement and fill in fields. Answer each field with words "
          "from its list only. Copy numbers exactly as written in the requirement.")
DEFECT_NAMES = ["open", "short", "mouse bite", "spur", "spurious copper", "pin hole", "missing hole",
                "missing pad", "hole breakout", "scratch", "foreign object", "any defect", "none"]  # none = the requirement only sets the IPC class
NAME_TO_CLS = {v: k for k, v in DISPLAY.items()}
MEASURE_NAMES = ["always", "size", "width loss", "spacing loss", "count"]
MEASURE_MAP = {"always": "always", "size": "size_mm", "width loss": "width_loss_pct",
               "spacing loss": "spacing_loss_pct", "count": "count"}
UNIT_NAMES = ["mm", "mil", "micron", "percent", "none"]
CMP_NAMES = ["more than", "at least"]
ACTION_NAMES = ["reject", "rework", "review", "accept"]

# Worked examples are OFF by default: both 1B and 3B copied the examples' answers (3B, defect field, tuning
# set: 13/20 right with examples, 17/20 without; llm/probe_spec.py).
FEWSHOT = True
# The class-only example comes FIRST: the last example before the real question biases small models.
EXAMPLES = [
    ("Build to Class 2.", [("Defect", "none"), ("Class number", "2")]),
    ("Copper splashes of 3 mil or more must be removed.",
     [("Defect", "spurious copper"), ("Measure", "size"), ("Number as written", "3"),
      ("Unit", "mil"), ("Comparison", "at least"), ("Action", "rework")]),
    ("Any break in a trace is a reject.",
     [("Defect", "open"), ("Measure", "always"), ("Action", "reject")]),
    ("Edge nicks may remove up to 25% of the line width.",
     [("Defect", "mouse bite"), ("Measure", "width loss"), ("Number as written", "25"),
      ("Unit", "percent"), ("Comparison", "more than"), ("Action", "reject")]),
]


@dataclass
class CompileResult:
    rule: Rule | None
    answers: dict = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    generated_tokens: int = 0
    decode_tps: list = field(default_factory=list)


def _lists(order_rng: random.Random | None):
    def sh(xs):
        xs = list(xs)
        if order_rng is not None:
            order_rng.shuffle(xs)
        return xs
    return {"Defect": sh(DEFECT_NAMES), "Measure": sh(MEASURE_NAMES), "Unit": sh(UNIT_NAMES),
            "Comparison": sh(CMP_NAMES), "Action": sh(ACTION_NAMES)}


def build_prompt(clause: str, lists: dict) -> str:
    fields = (
        "Fill in the fields below one per line, answering only the field you are asked.\nFields:\n"
        f"Defect: one of {', '.join(lists['Defect'])} (none = the requirement names no defect and only "
        "sets the IPC class)\n"
        "Class number: 1, 2 or 3 (only when Defect is none)\n"
        f"Measure: one of {', '.join(lists['Measure'])} "
        "(always = no number given; size = defect size; width loss / spacing loss = percent reduction of "
        "conductor width / spacing; count = how many on one board)\n"
        "Number as written: the number in the requirement (only when Measure is not always)\n"
        f"Unit: one of {', '.join(lists['Unit'])}\n"
        f"Comparison: one of {', '.join(lists['Comparison'])} "
        "(more than = above the number, also for 'up to X is acceptable' or 'shall not exceed X'; "
        "at least = the number itself counts, 'X or more')\n"
        f"Action: one of {', '.join(lists['Action'])} (scrap = reject; repair or remove = rework; "
        "hold, engineer or MRB = review)\n")
    if FEWSHOT and EXAMPLES:
        ex = "\n\n".join(f"Requirement: \"{t}\"\n" + "\n".join(f"{k}: {v}" for k, v in a) for t, a in EXAMPLES)
        user = f"{fields}\nExamples:\n\n{ex}\n\nNow this one.\nRequirement: \"{clause}\""
    else:
        user = f"{fields}\nRequirement: \"{clause}\""
    return (HDR + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


LABEL_STEMS = ("defect", "number", "measure", "unit", "comparison", "action")  # no listed answer contains these


def _line(text: str) -> str:
    """The answer to one field: its first line, cut where the model starts writing the NEXT field
    (without examples it often runs on: '3measure: count')."""
    t = text.split("\n")[0].lower()
    cut = [i for i in (t.find(st, 1) for st in LABEL_STEMS) if i > 0]
    if cut:
        t = t[:min(cut)]
    return t.strip().strip(":").strip(".").strip()


def _field_done(s: str) -> bool:
    """Stop at the end of the line, or as soon as the model starts the next field's label."""
    low = s.lower()
    return "\n" in s or any(low.find(st, 1) > 0 for st in LABEL_STEMS)


def _pick(ans: str, names) -> str | None:
    a = ans.lower()
    hits = [n for n in sorted(names, key=len, reverse=True) if n in a]
    return hits[0] if hits else None


def map_defect(ans: str) -> str | None:
    p = _pick(ans, DEFECT_NAMES)
    if p:
        return NAME_TO_CLS[p]
    if "any" in ans.lower():
        return "any"
    ds = defects_mentioned(ans)
    return sorted(ds)[0] if len(ds) == 1 else None


def map_action(ans: str) -> str | None:
    a = ans.lower()
    for k, ws in (("reject", ("reject", "scrap")), ("rework", ("rework", "repair", "remove")),
                  ("review", ("review", "hold", "mrb", "engineer")), ("accept", ("accept",))):
        if any(w in a for w in ws):
            return k
    return None


def parse_number(ans: str) -> float | None:
    ns = numbers_in(ans)
    return ns[0][0] if ns else None


def assemble(ans: dict) -> tuple[Rule | None, str | None]:
    """Field answers -> Rule (pure function; unit-tested without the model)."""
    if ans.get("Defect", "").startswith("none"):
        # exactly one class number ("1, 2, 3" is the model echoing the list, not an answer)
        m = re.fullmatch(r"\s*(?:class\s*)?([123])\s*", ans.get("Class number", ""))
        return (Rule(kind="class", customer_class=int(m.group(1)), action="reject"), None) if m \
            else (None, "no single class number")
    d = map_defect(ans.get("Defect", ""))
    if d is None:
        return None, "defect not recognised"
    mp = _pick(ans.get("Measure", ""), MEASURE_NAMES)
    if mp is None:
        return None, "measure not recognised"
    measure = MEASURE_MAP[mp]
    action = map_action(ans.get("Action", ""))
    if action is None:
        return None, "action not recognised"
    if measure == "always":
        return Rule(defect=d, measure="always", action=action), None
    num = parse_number(ans.get("Number as written", ""))
    if num is None:
        return None, "no number"
    unit = _pick(ans.get("Unit", ""), UNIT_NAMES)
    unit = {"percent": "%", "none": None, None: None}.get(unit, unit)
    if measure == "size_mm" and unit in (None, "%"):
        unit = "mm" if unit is None else unit
    value = to_canonical(num, unit, measure)
    if value is None:
        return None, f"unit {unit} does not fit measure {measure}"
    cmp_ = _pick(ans.get("Comparison", ""), CMP_NAMES)
    op = ">=" if cmp_ == "at least" else ">"
    return Rule(defect=d, measure=measure, op=op, value=value, action=action), None


class LlamaSpecCompiler:
    def __init__(self, backend, max_field_tokens: int = 8):
        self.be = backend
        self.n = max_field_tokens

    def compile(self, clause: str, order_seed: int | None = None) -> CompileResult:
        lists = _lists(random.Random(order_seed) if order_seed is not None else None)
        t0 = time.perf_counter()
        res = CompileResult(None)
        stop = _field_done
        self.be.reset()
        text = build_prompt(clause, lists)
        ans = {}

        def ask(label, prefix_nl=True):
            nonlocal text
            g = self.be.cont(text + f"{label}:", self.n, stop=stop)
            text = ""
            a = _line(g.text)
            ans[label] = a
            res.generated_tokens += g.n_generated or 0
            if res.prompt_tokens is None:
                res.prompt_tokens = g.prompt_tokens
            if g.decode_tps:
                res.decode_tps.append(g.decode_tps)
            # the generated tokens are ALREADY in the KV cache: never feed the answer back (that duplicated
            # every answer in the model's context and made it repeat itself); only close the line if
            # generation stopped before its newline
            text = "" if "\n" in g.text else "\n"
            return a

        if ask("Defect").startswith("none"):
            ask("Class number")
        else:
            if MEASURE_MAP.get(_pick(ask("Measure"), MEASURE_NAMES) or "", "always") != "always":
                ask("Number as written")
                ask("Unit")
                ask("Comparison")
            ask("Action")
        res.answers = ans
        res.rule, res.error = assemble(ans)
        res.latency_ms = (time.perf_counter() - t0) * 1e3
        return res


NUM_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")
WORD_NUMBERS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def numbers_as_written(clause: str) -> list[str]:
    """Every number the clause actually contains, as written (digits first, then number words)."""
    out = list(dict.fromkeys(m.group(0) for m in NUM_TOKEN_RE.finditer(clause)))
    low = clause.lower()
    out += [w for w in WORD_NUMBERS if re.search(rf"\b{w}\b", low)]
    return out


CLASS_NUM_RE = re.compile(r"\bclass\s*([123])\b", re.I)


def grounded_numbers(clause: str) -> list[tuple[str, str | None]]:
    """(number as written, unit) for every number in the clause that is not an IPC class number.
    unit: 'mm' | 'mil' | 'micron' | '%' | None (a bare count)."""
    from .spec import NUM_RE
    class_spans = [m.span(1) for m in CLASS_NUM_RE.finditer(clause)]
    out, seen = [], set()
    for m in NUM_RE.finditer(clause):
        if any(a0 <= m.start(1) < b0 for a0, b0 in class_spans):
            continue
        val, unit = numbers_in(m.group(0))[0]
        key = (m.group(1), unit)
        if key not in seen:
            seen.add(key)
            out.append(key)
    low = clause.lower()
    for w in WORD_NUMBERS:
        if re.search(rf"\b{w}\b", low) and (w, None) not in seen:
            out.append((w, None))
    return out


UNIT_MEASURES = {"%": ["width loss", "spacing loss"], "mm": ["size"], "mil": ["size"], "micron": ["size"],
                 None: ["count"]}
UNIT_NAME = {"%": "percent", "mm": "mm", "mil": "mil", "micron": "micron", None: "none"}


class ConstrainedSpecCompiler:
    """Same fields and prompt as LlamaSpecCompiler, but every answer is chosen by grammar-constrained
    decoding (llm_logits.LogitLlama.choose), so the model cannot answer off the list.

    grounded=False  the lists are the full vocabularies (and the numbers written in the clause).
    grounded=True   the clause's own words narrow the lists before the model chooses:
                      defect    only defects the clause names (full list if it names none or several ways)
                      measure   only measures that fit a number in the clause (25% -> width/spacing loss,
                                0.1 mm / 3 mil -> size, a bare number -> count; no number -> always)
                      number    only numbers whose unit fits the chosen measure
                      unit      the unit written next to the chosen number (not asked)
                    The model still decides everything the words leave open: which of the named defects,
                    width or spacing loss, which number, more-than or at-least, and the action.
    Each choice keeps its margin (how decisively it won) in answers['_margins']."""

    def __init__(self, backend, grounded: bool = False):
        self.be = backend
        self.grounded = grounded

    def compile(self, clause: str, order_seed: int | None = None) -> CompileResult:
        lists = _lists(random.Random(order_seed) if order_seed is not None else None)
        t0 = time.perf_counter()
        res = CompileResult(None)
        be = self.be
        be.reset()
        n0 = be.n_forward
        be.feed(build_prompt(clause, lists))
        res.prompt_tokens = be.pos
        ans, margins = {}, {}
        first = True

        def ask(label, options):
            nonlocal first
            be.feed(f"{label}:" if first else f"\n{label}:")
            first = False
            if len(options) == 1:          # nothing to decide: write it into the context and move on
                be.feed(" " + options[0])
                ans[label], margins[label] = options[0], 99.0
                return options[0]
            opt, m = be.choose([" " + o for o in options])
            ans[label], margins[label] = opt.strip(), round(float(min(m, 99.0)), 3)
            return ans[label]

        gnums = grounded_numbers(clause)
        defects = lists["Defect"]
        if self.grounded:
            named = [DISPLAY[d] for d in sorted(defects_mentioned(clause)) if d in DISPLAY]
            if named:
                defects = [d for d in lists["Defect"] if d in named]
            elif not gnums and CLASS_NUM_RE.search(clause):
                defects = ["none", "any defect"]
        if ask("Defect", defects) == "none":
            cls = list(dict.fromkeys(CLASS_NUM_RE.findall(clause))) if self.grounded else []
            ask("Class number", cls or ["1", "2", "3"])
        else:
            if self.grounded:
                fit = [m for m in lists["Measure"] if any(m in UNIT_MEASURES[u] for _, u in gnums)]
                measures = fit or ["always"]
            else:
                measures = lists["Measure"] if gnums else ["always"]
            if MEASURE_MAP[ask("Measure", measures)] != "always":
                cands = [(n, u) for n, u in gnums if not self.grounded or ans["Measure"] in UNIT_MEASURES[u]]
                num = ask("Number as written", list(dict.fromkeys(n for n, _ in cands)))
                unit = next(u for n, u in cands if n == num)
                if self.grounded:
                    ask("Unit", [UNIT_NAME[unit]])
                else:
                    ask("Unit", lists["Unit"])
                ask("Comparison", lists["Comparison"])
            ask("Action", lists["Action"])
        ans = {k: (WORD_NUM_VALUE.get(v, v) if k == "Number as written" else v) for k, v in ans.items()}
        res.answers = {**ans, "_margins": margins, "_grounded": self.grounded}
        res.rule, res.error = assemble(ans)
        res.generated_tokens = be.n_forward - n0 - 1
        res.latency_ms = (time.perf_counter() - t0) * 1e3
        return res


WORD_NUM_VALUE = {w: str(i + 1) for i, w in enumerate(WORD_NUMBERS)}

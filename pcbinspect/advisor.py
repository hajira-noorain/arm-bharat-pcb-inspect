"""Operator advice: Llama (ExecuTorch) constrained to the knowledge base, with a deterministic fallback.

Step-wise structured generation (the default with the ExecuTorch runtime):
  1. the board context is prefilled ONCE: each defect with its cause/action options, numbered in a
     per-board shuffled order (so a 'always pick option 1' habit cannot look like a good choice)
  2. for every defect, most severe first, the advisor appends '{"defect": "Dk", "cause": ' and the
     model fills in only two numbers (stopped at '}') — a constrained, measurable decision
  3. finally '"summary": "' and the model writes one or two sentences (stopped at the closing quote)
  4. numbers -> KB IDs through that board's option order; the grounding checker validates the result
     (summary numbers and defect types must come from the record); anything invalid -> template advice

Who decides what is stated honestly in the report: the ORDER of steps (severity) and COVERAGE (every
defect gets a step) are by construction; the CHOICE of cause/action and the SUMMARY are Llama's.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .grounding import GroundingResult, check
from .kb import KnowledgeBase, default_kb

SEV_RANK = {"critical": 0, "major": 1, "minor": 2, None: 3}
MAX_LLM_DEFECTS = 5          # keep the prompt short on a Pi; the rest get template advice
PROMPT_STYLE = "numbered"     # 'numbered' (default) or 'ids' (v1, kept for the comparison)
HDR = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
SYSTEM = ("You are the quality assistant on a PCB inspection station. Choose only from the listed "
          "options. Reply with JSON only.")
# cause number, then the action number; whatever follows (usually '}') is not required
CHOICE_RE = re.compile(r'^\s*"?(\d+)"?\s*,\s*"action"\s*:\s*"?(\d+)"?\s*\}?')
CHOICE_DONE_RE = re.compile(r'"action"\s*:\s*"?\d+\D')   # second number complete -> stop


def _choice_done(text: str) -> bool:
    return bool(CHOICE_DONE_RE.search(text))


@dataclass
class GenResult:
    text: str
    latency_ms: float
    prefill_tps: float | None = None
    decode_tps: float | None = None
    n_generated: int | None = None
    prompt_tokens: int | None = None


class LLMBackend(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult: ...


@dataclass
class Advice:
    backend: str                  # 'llama-executorch' | 'template'
    model: str | None
    summary: str
    steps: list[dict]             # rendered steps with KB text
    grounded: bool
    rejected_reasons: list[str] = field(default_factory=list)
    filled_from_template: list[str] = field(default_factory=list)
    raw_output: str | None = None
    gen: GenResult | None = None
    choices: list[dict] = field(default_factory=list)   # per defect: shown positions chosen (eval)

    def to_json(self) -> dict:
        return {"summary": self.summary, "steps": self.steps, "backend": self.backend,
                "model": self.model, "filled_from_template": self.filled_from_template}


def _sorted_defects(record: dict) -> list[dict]:
    return sorted(record["defects"], key=lambda d: (SEV_RANK.get(d.get("severity"), 3),
                                                    -(d.get("confidence") or 0)))


def history_numbers(history: dict[str, int] | None, window: int = 20) -> list[float]:
    """Numbers the prompt states about history: allowed in the summary by the grounding check."""
    if not history:
        return []
    return [float(window)] + [float(v) for v in history.values() if v]


def option_orders(record: dict, kb: KnowledgeBase, shuffle: bool = True) -> dict[str, tuple[list, list]]:
    """Per-defect order in which cause/action options are shown (seeded by board + defect)."""
    out = {}
    for d in _sorted_defects(record)[:MAX_LLM_DEFECTS]:
        e = kb.entry(d["type"])
        c, a = [x["id"] for x in e["causes"]], [x["id"] for x in e["actions"]]
        if shuffle:
            rng = random.Random(f"{record.get('board_id')}|{d['id']}")
            rng.shuffle(c)
            rng.shuffle(a)
        out[d["id"]] = (c, a)
    return out


def build_prompt(record: dict, history: dict[str, int] | None, kb: KnowledgeBase,
                 window: int = 20, style: str | None = None,
                 orders: dict[str, tuple[list, list]] | None = None,
                 prefix: str | None = None) -> str:
    """Compact Llama 3.2 Instruct chat prompt (prefill dominates on a Pi, so every token counts).

    history = {defect_type: number of the last `window` boards on which it appeared}.
    """
    style = style or PROMPT_STYLE
    if style == "numbered" and orders is None:
        orders = option_orders(record, kb, shuffle=False)
    defects = _sorted_defects(record)[:MAX_LLM_DEFECTS]
    lines = []
    for d in defects:
        e = kb.entry(d["type"])
        size = f", {d['size_mm']} mm" if d.get("size_mm") is not None else ""
        lines.append(f"{d['id']}: {d['type'].replace('_', ' ')}, {d.get('severity')}{size}, "
                     f"{d.get('location')}, confidence {d.get('confidence')}")
        if style == "numbered":
            cids, aids = orders[d["id"]]
            lines.append("  causes: " + "; ".join(f"{i + 1} {kb.by_id[x]['short']}" for i, x in enumerate(cids)))
            lines.append("  actions: " + "; ".join(f"{i + 1} {kb.by_id[x]['short']}" for i, x in enumerate(aids)))
        else:
            lines.append("  causes: " + " | ".join(f"{c['id']} {c['short']}" for c in e["causes"]))
            lines.append("  actions: " + " | ".join(f"{a['id']} {a['short']}" for a in e["actions"]))
    hist = ""
    if history:
        parts = [f"{k.replace('_', ' ')} on {v} of the last {window} boards"
                 for k, v in sorted(history.items()) if v]
        if parts:
            hist = "History: " + ", ".join(parts) + ".\n"
    if style == "numbered":
        pick = "For each defect pick one cause number and one action number from its own lists. "
        fmt = 'JSON format: {"steps": [{"defect": "D1", "cause": 1, "action": 1}], "summary": "..."}'
    else:
        pick = "For each defect pick one cause ID and one action ID from its own lists. "
        fmt = ('JSON format: {"steps": [{"defect": "D1", "cause": "<cause ID>", "action": "<action ID>"}], '
               '"summary": "..."}')
    user = (
        f"Board {record['board_id']} verdict {record['verdict']}. Defects:\n" + "\n".join(lines) +
        "\n" + hist + pick +
        "A defect that keeps repeating points to a process cause; a one-off points to handling or "
        "a local cause. Then a summary of one or two sentences for the operator using only the "
        "facts above.\n" + fmt
    )
    return (HDR + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +
            (prefix if prefix is not None else assistant_prefix(record, style)))


def assistant_prefix(record: dict, style: str | None = None) -> str:
    """Start of the answer for the single-shot path: the first thing the model produces is the cause
    choice for the most severe defect."""
    style = style or PROMPT_STYLE
    first = _sorted_defects(record)[0]["id"]
    return '{"steps": [{"defect": "' + first + '", "cause": ' + ('' if style == "numbered" else '"')


def _join(already: str, want: str) -> str:
    """The stop token may already contain the start of what we append next (e.g. '},' or '}],').
    Drop that overlap so the model's context stays clean JSON."""
    a = already.strip()
    w = want
    for ch in a:
        if ch.isspace():
            continue
        ws = w.lstrip()
        if ws.startswith(ch):
            w = ws[1:]
        else:
            break
    return w


def template_advice(record: dict, kb: KnowledgeBase) -> dict:
    defects = _sorted_defects(record)
    if not defects:
        return {"summary": "No defects found.", "steps": []}
    steps = []
    for d in defects:
        e = kb.entry(d["type"])
        steps.append({"defect": d["id"], "cause": e["causes"][0]["id"],
                      "action": e["actions"][0]["id"]})
    top = defects[0]
    n = len(defects)
    summary = (f"{n} defect{'s' if n > 1 else ''} found. Most serious: "
               f"{kb.display_name(top['type']).lower()} ({top.get('severity')}) at "
               f"{top.get('location') or 'the marked area'}.")
    return {"summary": summary, "steps": steps}


def render(advice: dict, record: dict, kb: KnowledgeBase) -> list[dict]:
    by_id = {d["id"]: d for d in record["defects"]}
    out = []
    for s in advice["steps"]:
        d = by_id[s["defect"]]
        e = kb.entry(d["type"])
        cause = kb.by_id[s["cause"]]
        out.append({
            "defect": s["defect"], "type": d["type"], "name": e["name"],
            "severity": d.get("severity"), "location": d.get("location"),
            "cause_id": s["cause"], "cause": cause["text"],
            "process_step": kb.process_steps.get(cause.get("step", ""), ""),
            "action_id": s["action"], "action": kb.text(s["action"]),
        })
    return out


class Advisor:
    def __init__(self, backend: LLMBackend | None, kb: KnowledgeBase | None = None,
                 max_new_tokens: int = 160, summary_tokens: int = 90):
        self.backend = backend
        self.kb = kb or default_kb()
        self.max_new_tokens = max_new_tokens
        self.summary_tokens = summary_tokens

    def _template(self, record, reasons, model=None, raw=None, gen=None) -> Advice:
        t = template_advice(record, self.kb)
        return Advice("template", model, t["summary"], render(t, record, self.kb), True,
                      reasons, [], raw, gen)

    def advise(self, record: dict, history: dict[str, int] | None = None) -> Advice:
        if not record["defects"]:
            t = template_advice(record, self.kb)
            return Advice("template", None, t["summary"], [], True)
        if self.backend is None:
            return self._template(record, ["no LLM backend loaded"])
        try:
            if hasattr(self.backend, "cont"):
                return self._advise_stepwise(record, history)
            return self._advise_single(record, history)
        except Exception as exc:  # runtime failure must never block the inspection
            return self._template(record, [f"LLM error: {exc!r}"])

    # ---------------------------------------------------------------- step-wise (default)
    def _advise_stepwise(self, record: dict, history: dict[str, int] | None) -> Advice:
        be, kb = self.backend, self.kb
        defects = _sorted_defects(record)[:MAX_LLM_DEFECTS]
        orders = option_orders(record, kb, shuffle=True)
        start = '{"steps": ['
        prompt = build_prompt(record, history, kb, style="numbered", orders=orders, prefix=start)
        t0 = time.perf_counter()
        be.reset()
        raw, carry = start, ""
        steps, choices, reasons = [], [], []
        gens: list[GenResult] = []
        for i, d in enumerate(defects):
            want = ("" if i == 0 else ", ") + '{"defect": "' + d["id"] + '", "cause": '
            send = (prompt if i == 0 else "") + _join(carry, want)
            g = be.cont(send, 14, stop=_choice_done)
            gens.append(g)
            raw += _join(carry, want) + g.text
            m = CHOICE_RE.match(g.text)
            if not m:
                reasons.append(f"{d['id']}: choice not in the expected form ({g.text.strip()[:40]!r})")
                break
            c, a = int(m.group(1)), int(m.group(2))
            cids, aids = orders[d["id"]]
            if not (1 <= c <= len(cids) and 1 <= a <= len(aids)):
                reasons.append(f"{d['id']}: option {c}/{a} out of range")
                break
            steps.append({"defect": d["id"], "cause": cids[c - 1], "action": aids[a - 1]})
            choices.append({"defect": d["id"], "type": d["type"], "cause_pos": c, "n_causes": len(cids),
                            "action_pos": a, "n_actions": len(aids), "cause": cids[c - 1]})
            carry = g.text[m.end():]
        summary = ""
        if not reasons:
            want = '], "summary": "'
            g = be.cont(_join(carry, want), self.summary_tokens, stop='"')
            gens.append(g)
            raw += _join(carry, want) + g.text
            summary = g.text.split('"')[0].strip()
        total = GenResult(
            text=raw, latency_ms=(time.perf_counter() - t0) * 1e3,
            prefill_tps=gens[0].prefill_tps if gens else None,
            decode_tps=_mean([g.decode_tps for g in gens if g.decode_tps]),
            n_generated=sum(g.n_generated or 0 for g in gens),
            prompt_tokens=gens[0].prompt_tokens if gens else None)
        if reasons:
            return self._template(record, reasons, be.model, raw, total)
        candidate = json.dumps({"steps": steps, "summary": summary})
        res: GroundingResult = check(candidate, record, kb, extra_numbers=history_numbers(history))
        if not res.ok:
            adv = self._template(record, res.reasons, be.model, raw, total)
            adv.choices = choices
            return adv
        covered = {s["defect"] for s in res.advice["steps"]}
        extra = [s for s in template_advice(record, kb)["steps"] if s["defect"] not in covered]
        res.advice["steps"].extend(extra)
        return Advice(be.name, be.model, res.advice["summary"], render(res.advice, record, kb), True,
                      [], [s["defect"] for s in extra], raw, total, choices)

    # ---------------------------------------------------------------- single-shot (fallback)
    def _advise_single(self, record: dict, history: dict[str, int] | None) -> Advice:
        prompt = build_prompt(record, history, self.kb)
        gen = self.backend.generate(prompt, self.max_new_tokens)
        raw = assistant_prefix(record) + gen.text
        res = check(raw, record, self.kb, extra_numbers=history_numbers(history))
        if not res.ok:
            return self._template(record, res.reasons, self.backend.model, raw, gen)
        covered = {s["defect"] for s in res.advice["steps"]}
        extra = [s for s in template_advice(record, self.kb)["steps"] if s["defect"] not in covered]
        res.advice["steps"].extend(extra)
        return Advice(self.backend.name, self.backend.model, res.advice["summary"],
                      render(res.advice, record, self.kb), True, [],
                      res.filled_from_template + [s["defect"] for s in extra], raw, gen)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None

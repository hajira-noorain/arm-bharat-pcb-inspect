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


OPTION_ORDER = "evidence"   # production: 'evidence'; evaluation of the model itself: 'shuffle'


def option_orders(record: dict, kb: KnowledgeBase, shuffle: bool = True,
                  history: dict[str, int] | None = None, mode: str | None = None) -> dict[str, tuple[list, list]]:
    """Per-defect order in which cause/action options are shown.

    'shuffle'  seeded random order: used to MEASURE the model, so a first-option habit cannot look
               like reasoning (measured: Llama 3.2 1B/3B choose causes at about chance level)
    'evidence' the process rule ranks causes: a defect repeating on the line puts process causes
               (etching, imaging, drilling, plating) first, a one-off puts handling causes first.
               Used in production, because the model alone does not rank causes better than chance.
    'kb'       knowledge-base order
    """
    mode = mode or ("shuffle" if shuffle else "kb")
    out = {}
    for d in _sorted_defects(record)[:MAX_LLM_DEFECTS]:
        e = kb.entry(d["type"])
        causes, a = list(e["causes"]), [x["id"] for x in e["actions"]]
        if mode == "shuffle":
            rng = random.Random(f"{record.get('board_id')}|{d['id']}")
            rng.shuffle(causes)
            rng.shuffle(a)
        elif mode == "evidence":
            repeating = bool(history) and history.get(d["type"], 0) >= 3
            causes.sort(key=lambda c: (c.get("step") == "handling") == repeating)   # stable sort
        out[d["id"]] = ([c["id"] for c in causes], a)
    return out


FEWSHOT = False   # worked example in the prompt; switched on only if the evaluation shows it helps
EXAMPLE_TYPES = ("open", "short", "spurious_copper", "unknown_anomaly")   # have process + handling causes


def _worked_example(record: dict, history: dict | None, kb: KnowledgeBase, window: int = 20) -> str:
    """One worked example matched to the situation (repeating vs one-off), on a defect type that is
    NOT on this board, so the model learns the reasoning rather than copying a position."""
    present = {d["type"] for d in record["defects"]}
    ex_type = next((t for t in EXAMPLE_TYPES if t not in present), None)
    if ex_type is None:
        return ""
    repeating = bool(history) and any(t in present for t in history)
    causes = list(kb.entry(ex_type)["causes"])
    random.Random(f"example|{ex_type}").shuffle(causes)
    want = next(i for i, c in enumerate(causes, 1)
                if (c.get("step") != "handling") == repeating)
    hist = (f" History: {ex_type.replace('_', ' ')} on 6 of the last {window} boards." if repeating
            else " It has not been seen on recent boards.")
    why = "it keeps repeating, so the process is the likely cause" if repeating else \
        "it is a one-off, so a local or handling cause is likely"
    return ("Worked example from a different board (do not mention it in your answer): D9: "
            f"{ex_type.replace('_', ' ')}. causes: " +
            "; ".join(f"{i} {c['short']}" for i, c in enumerate(causes, 1)) + f".{hist} "
            f"Answer cause {want} because {why}.\n")


def build_prompt(record: dict, history: dict[str, int] | None, kb: KnowledgeBase,
                 window: int = 20, style: str | None = None,
                 orders: dict[str, tuple[list, list]] | None = None,
                 prefix: str | None = None, fewshot: bool | None = None) -> str:
    """Compact Llama 3.2 Instruct chat prompt (prefill dominates on a Pi, so every token counts).

    history = {defect_type: number of the last `window` boards on which it appeared}.
    """
    style = style or PROMPT_STYLE
    fewshot = FEWSHOT if fewshot is None else fewshot
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
    example = _worked_example(record, history, kb) if (fewshot and style == "numbered") else ""
    user = (
        f"Board {record['board_id']} verdict {record['verdict']}. Defects:\n" + "\n".join(lines) +
        "\n" + hist + example + pick +
        "A defect that keeps repeating points to a process cause; a one-off points to handling or "
        "a local cause. Then a summary of one or two sentences for the operator using only the "
        "facts above.\n" + fmt
    )
    return (HDR + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +
            (prefix if prefix is not None else assistant_prefix(record, style)))


LLAMA_BACKEND = "llama-executorch"     # what an accepted Llama answer is logged as, whichever runner made it
SCORED_SYSTEM = ("You are the quality assistant on a PCB inspection station. For each defect you name the "
                 "most likely cause and the best next action, using only the options listed for it.")


def build_scored_prompt(record: dict, history: dict[str, int] | None, kb: KnowledgeBase,
                        orders: dict[str, tuple[list, list]], window: int = 20) -> str:
    """Prompt for the scored path. Each cause carries its process step in brackets, and each defect line
    says whether that defect has been repeating, so the rule in the last paragraph can be applied."""
    lines = []
    for d in _sorted_defects(record)[:MAX_LLM_DEFECTS]:
        size = f", {d['size_mm']} mm" if d.get("size_mm") is not None else ""
        seen = (history or {}).get(d["type"], 0)
        rep_txt = (f"REPEATING: on {seen} of the last {window} boards" if seen >= 3 else
                   "ONE-OFF: not on other recent boards")
        cids, aids = orders[d["id"]]
        lines.append(f"{d['id']}: {d['type'].replace('_', ' ')}, {d.get('severity')}{size}, {d.get('location')}, "
                     f"confidence {d.get('confidence')}. {rep_txt}.")
        lines.append("  causes: " + "; ".join(
            f"{kb.by_id[x]['short']} ({kb.by_id[x].get('step', 'process')})" for x in cids))
        lines.append("  actions: " + "; ".join(kb.by_id[x]["short"] for x in aids))
    user = (f"Board {record['board_id']} verdict {record['verdict']}. Defects:\n" + "\n".join(lines) +
            "\nRule: a REPEATING defect comes from the process (imaging, etching, drilling, plating, assembly), "
            "so choose a cause from one of those steps. A ONE-OFF defect comes from handling or a local "
            "accident, so choose a handling cause if one is listed.\n"
            "Answer one line per item, copying the option text, then one line 'Summary:' with one or two "
            "sentences for the operator that use only the facts above.")
    return (HDR + SCORED_SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


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
                 max_new_tokens: int = 160, summary_tokens: int = 90, fewshot: bool | None = None,
                 option_order: str | None = None, score_norm: str = "sum"):
        self.score_norm = score_norm      # scored path: 'sum' (log-probability of the whole option) or 'mean' per token
        self.last_forwards = 0
        self.fewshot = fewshot
        self.option_order = option_order or OPTION_ORDER
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
            if hasattr(self.backend, "score"):
                return self._advise_scored(record, history)
            if hasattr(self.backend, "cont"):
                return self._advise_stepwise(record, history)
            return self._advise_single(record, history)
        except Exception as exc:  # runtime failure must never block the inspection
            return self._template(record, [f"LLM error: {exc!r}"])

    # ---------------------------------------------------------------- scored choices (logits backend)
    def _advise_scored(self, record: dict, history: dict[str, int] | None, window: int = 20) -> Advice:
        """Llama through llm_logits.LogitLlama. Nothing it outputs can be malformed:
          cause / action  the option whose TEXT is the most probable continuation (full-sequence score,
                          including the probability that the answer ends there). No option numbers, so
                          the habit of answering "1" has nothing to hold on to.
          summary         free text, but a number that is not in the findings cannot be written.
        The grounding checker still runs on the result."""
        be, kb = self.backend, self.kb
        defects = _sorted_defects(record)[:MAX_LLM_DEFECTS]
        orders = option_orders(record, kb, history=history, mode=self.option_order)
        t0 = time.perf_counter()
        be.reset()
        n0 = be.n_forward
        be.feed(build_scored_prompt(record, history, kb, orders, window))
        n_prompt, t1 = be.pos, time.perf_counter()
        steps, choices, lines = [], [], []
        for i, d in enumerate(defects):
            cids, aids = orders[d["id"]]
            picked = {}
            for what, ids in (("cause", cids), ("action", aids)):
                be.feed(("" if not lines else "\n") + f"{d['id']} {what}:")
                opts = [" " + kb.by_id[x]["short"] for x in ids]
                sc = be.score(opts, end="\n")
                val = [lp / n if self.score_norm == "mean" else lp for lp, n in sc]
                j = max(range(len(ids)), key=lambda k: val[k])
                be.feed(opts[j])
                picked[what] = (j, ids[j])
                lines.append(f"{d['id']} {what}:{opts[j]}")
            steps.append({"defect": d["id"], "cause": picked["cause"][1], "action": picked["action"][1]})
            choices.append({"defect": d["id"], "type": d["type"], "cause_pos": picked["cause"][0] + 1,
                            "n_causes": len(cids), "action_pos": picked["action"][0] + 1,
                            "n_actions": len(aids), "cause": picked["cause"][1]})
        be.feed("\nSummary:")
        from .grounding import number_filter
        summary = be.write(self.summary_tokens, stop=("\n",),
                           allow=number_filter(record, history_numbers(history))).strip().strip('"').strip()
        t2 = time.perf_counter()
        n_gen = be.pos - n_prompt
        raw = "\n".join(lines) + "\nSummary: " + summary
        total = GenResult(raw, (t2 - t0) * 1e3, n_prompt / max(t1 - t0, 1e-6), n_gen / max(t2 - t1, 1e-6),
                          n_gen, n_prompt)
        self.last_forwards = be.n_forward - n0
        if not summary:
            adv = self._template(record, ["empty summary"], be.model, raw, total)
            adv.choices = choices
            return adv
        res: GroundingResult = check(json.dumps({"steps": steps, "summary": summary}), record, kb,
                                     extra_numbers=history_numbers(history))
        if not res.ok:
            adv = self._template(record, res.reasons, be.model, raw, total)
            adv.choices = choices
            return adv
        covered = {s["defect"] for s in res.advice["steps"]}
        extra = [s for s in template_advice(record, kb)["steps"] if s["defect"] not in covered]
        res.advice["steps"].extend(extra)
        return Advice(LLAMA_BACKEND, be.model, res.advice["summary"], render(res.advice, record, kb), True,
                      [], [s["defect"] for s in extra], raw, total, choices)

    # ---------------------------------------------------------------- step-wise (default)
    def _advise_stepwise(self, record: dict, history: dict[str, int] | None) -> Advice:
        be, kb = self.backend, self.kb
        defects = _sorted_defects(record)[:MAX_LLM_DEFECTS]
        orders = option_orders(record, kb, history=history, mode=self.option_order)
        start = '{"steps": ['
        prompt = build_prompt(record, history, kb, style="numbered", orders=orders, prefix=start,
                              fewshot=self.fewshot)
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

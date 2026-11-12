"""Operator advice: Llama (ExecuTorch) constrained to the knowledge base, with a deterministic fallback.

Flow:  record + KB options + recent history  ->  prompt  ->  Llama  ->  grounding check
       -> pass: render (KB text verbatim + Llama's choice/order/summary)
       -> fail: template advice (and the rejection reason is logged, which feeds the
                published hallucination-rate metric)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from .grounding import GroundingResult, check
from .kb import KnowledgeBase, default_kb

SEV_RANK = {"critical": 0, "major": 1, "minor": 2, None: 3}
MAX_LLM_DEFECTS = 5          # keep the prompt short on a Pi; the rest get template advice
ASSISTANT_PREFIX = '{"summary": "'


@dataclass
class GenResult:
    text: str
    latency_ms: float
    prefill_tps: float | None = None
    decode_tps: float | None = None
    n_generated: int | None = None


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

    def to_json(self) -> dict:
        return {"summary": self.summary, "steps": self.steps, "backend": self.backend,
                "model": self.model, "filled_from_template": self.filled_from_template}


def _sorted_defects(record: dict) -> list[dict]:
    return sorted(record["defects"], key=lambda d: (SEV_RANK.get(d.get("severity"), 3),
                                                    -(d.get("confidence") or 0)))


def build_prompt(record: dict, history: dict[str, int] | None, kb: KnowledgeBase,
                 window: int = 20) -> str:
    """Llama 3.2 Instruct chat prompt. history = {defect_type: count in last `window` boards}."""
    defects = _sorted_defects(record)[:MAX_LLM_DEFECTS]
    small = {**record, "defects": defects}
    lines = []
    for d in defects:
        e = kb.entry(d["type"])
        size = f", {d['size_mm']} mm" if d.get("size_mm") is not None else ""
        lines.append(f"{d['id']} ({d['type']}, {d.get('severity')}{size}, {d.get('location')}):")
        lines.append("  causes: " + "; ".join(f"{c['id']}: {c['text']}" for c in e["causes"]))
        lines.append("  actions: " + "; ".join(f"{a['id']}: {a['text']}" for a in e["actions"]))
    hist = ""
    if history:
        parts = [f"{k} on {v} of the last {window} boards" for k, v in sorted(history.items()) if v]
        if parts:
            hist = "Recent history for this design: " + ", ".join(parts) + ".\n"
    system = ("You are the quality assistant on a PCB inspection station. Use only the options "
              "given for each defect. Reply with JSON only.")
    user = (
        f"Inspection record:\n{json.dumps(small, separators=(',', ':'))}\n\n"
        "Options per defect:\n" + "\n".join(lines) + "\n\n" + hist +
        "Task: for each defect choose the single most likely cause and the most useful next "
        "action from its own options. A defect type that keeps repeating points to a process "
        "cause. Put the most severe defect first. Then write a summary of at most two sentences "
        "for the operator using only facts from the record.\n"
        'Reply in exactly this JSON format: {"summary": "...", "steps": [{"defect": "D1", '
        '"cause": "<cause id>", "action": "<action id>"}]}'
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + ASSISTANT_PREFIX
    )


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
            "defect": s["defect"],
            "type": d["type"],
            "name": e["name"],
            "severity": d.get("severity"),
            "location": d.get("location"),
            "cause_id": s["cause"],
            "cause": cause["text"],
            "process_step": kb.process_steps.get(cause.get("step", ""), ""),
            "action_id": s["action"],
            "action": kb.text(s["action"]),
        })
    return out


class Advisor:
    def __init__(self, backend: LLMBackend | None, kb: KnowledgeBase | None = None,
                 max_new_tokens: int = 160):
        self.backend = backend
        self.kb = kb or default_kb()
        self.max_new_tokens = max_new_tokens

    def advise(self, record: dict, history: dict[str, int] | None = None) -> Advice:
        if not record["defects"]:
            t = template_advice(record, self.kb)
            return Advice("template", None, t["summary"], [], True)

        if self.backend is None:
            t = template_advice(record, self.kb)
            return Advice("template", None, t["summary"], render(t, record, self.kb), True,
                          ["no LLM backend loaded"])

        prompt = build_prompt(record, history, self.kb)
        t0 = time.perf_counter()
        try:
            gen = self.backend.generate(prompt, self.max_new_tokens)
        except Exception as exc:  # runtime failure must never block the inspection
            t = template_advice(record, self.kb)
            return Advice("template", None, t["summary"], render(t, record, self.kb), True,
                          [f"LLM error: {exc!r}"])
        raw = ASSISTANT_PREFIX + gen.text
        res: GroundingResult = check(raw, record, self.kb)
        # defects beyond the prompt cap get template steps
        if res.ok:
            covered = {s["defect"] for s in res.advice["steps"]}
            tmpl = template_advice(record, self.kb)
            extra = [s for s in tmpl["steps"] if s["defect"] not in covered]
            res.advice["steps"].extend(extra)
            return Advice(self.backend.name, self.backend.model, res.advice["summary"],
                          render(res.advice, record, self.kb), True, [],
                          res.filled_from_template + [s["defect"] for s in extra], raw, gen)
        t = template_advice(record, self.kb)
        return Advice("template", self.backend.model, t["summary"], render(t, record, self.kb),
                      True, res.reasons, [], raw, gen)

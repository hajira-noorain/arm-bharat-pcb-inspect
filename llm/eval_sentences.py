"""How well does a Llama .pte write fill-in sentences for shift questions?

Test questions: the 20 held-out questions of llm/ask_questions.yaml (never used for training) and the six questions
recorded for the online demo, each with facts from generated rows (seeded), plus the demo's real rows for the six.
For every answer the station's own checker (pcbinspect/sentences.py) decides; the report counts each reason.

    python llm/eval_sentences.py --pte <model.pte> --tok <tokenizer.model> --out results/measured/sentences_<name>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.data.gen_sentences import _questions, _rows  # noqa: E402
from pcbinspect.ask import QUERIES, _no_digits, question_params  # noqa: E402
from pcbinspect.sentences import check_template, facts_for, nothing_to_say, render, template_prompt  # noqa: E402

DEMO = [  # the questions and rows recorded on the online demo (deploy/record_demo.py), where the 3B made mistakes
    ("Which defect is the most common this shift?", "defect_counts",
     [{"defect": "unknown_anomaly", "n": 28}, {"defect": "spurious_copper", "n": 25}, {"defect": "short", "n": 19}]),
    ("What is the pass rate today?", "yield", [{"boards": 28, "passed": 3, "failed": 0, "review": 25}]),
    ("How many boards went to a person for checking today?", "yield",
     [{"boards": 28, "passed": 3, "failed": 0, "review": 25}]),
    ("Split the defects from today into critical, major and minor", "severity_counts",
     [{"severity": "major", "n": 66}, {"severity": "critical", "n": 44}, {"severity": "minor", "n": 32}]),
    ("What happened with board B00005?", "board_history",
     [{"inspection": 5, "time": "2026-11-14 05:27:49", "verdict": "REVIEW", "defects": 6}]),
    ("Are there any open alerts?", "open_alerts",
     [{"time": "2026-11-14 05:30:00", "message": "short on 6 of the last 20 boards: check the etch line"}]),
]


def cases(seed: int = 31, per_question: int = 3) -> list[tuple]:
    rng = random.Random(seed)
    out = []
    for qid, qs in _questions("heldout").items():
        for q in qs:
            for _ in range(per_question):
                rows = _rows(qid, q, rng)
                params = {**QUERIES[qid]["params"], **question_params(q, qid)}
                facts = facts_for(qid, rows, params, q)
                if rows and not nothing_to_say(qid, facts):
                    out.append(("heldout", qid, q, facts))
    for q, qid, rows in DEMO:
        params = {**QUERIES[qid]["params"], **question_params(q, qid)}
        out.append(("demo", qid, q, facts_for(qid, rows, params, q)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    a = ap.parse_args()
    from pcbinspect.llm_logits import LogitLlama
    be = LogitLlama(a.pte, a.tok)
    rows, reasons = [], collections.Counter()
    t0 = time.perf_counter()
    for group, qid, q, facts in cases():
        be.reset()
        be.feed(template_prompt(q, facts))
        tpl = be.write(a.max_new_tokens, stop=("\n\n",), allow=_no_digits, nonempty=True).strip()
        bad = check_template(tpl, q, qid, facts)
        for r in bad:
            reasons[r.split(":")[0]] += 1
        rows.append({"group": group, "qid": qid, "question": q, "template": tpl, "ok": not bad, "reasons": bad,
                     "answer": render(tpl, facts) if not bad else None})
    summary = {}
    for g in ("heldout", "demo"):
        rs = [r for r in rows if r["group"] == g]
        summary[g] = {"n": len(rs), "accepted": sum(r["ok"] for r in rs) / max(len(rs), 1)}
    out = {"model": be.model, "seconds": round(time.perf_counter() - t0, 1), "summary": summary,
           "reasons": dict(reasons), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summary, "reasons": dict(reasons)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

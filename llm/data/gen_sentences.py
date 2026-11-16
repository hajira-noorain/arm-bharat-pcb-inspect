"""Training data for fill-in sentences (pcbinspect/sentences.py).

Each example is a question, the facts the station would fill in for it (from random rows, including zeros and
ties), and a correct fill-in sentence written from several hand-made phrasings. Every target passes the station's
own checker. The 20 held-out questions in llm/ask_questions.yaml are never used here; they are for evaluation.

    python -m llm.data.gen_sentences --n 3000 --out <dir>      # writes sentences_train.jsonl and sentences_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from pcbinspect.ask import QUERIES, question_params
from pcbinspect.sentences import (TYPE_WORDS, facts_for, named_types, nothing_to_say, required_fields,
                                  template_prompt)

ROOT = Path(__file__).resolve().parent.parent.parent
QIDS = list(QUERIES)
TYPES = list(TYPE_WORDS)
PLURAL = {"open": "opens", "short": "shorts", "mouse_bite": "mouse bites", "spur": "spurs",
          "spurious_copper": "copper islands", "pin_hole": "pinholes", "missing_hole": "missing holes",
          "missing_pad": "missing pads", "hole_breakout": "hole breakouts", "scratch": "scratches",
          "foreign_object": "foreign objects", "missing_part": "missing parts", "extra_part": "extra parts",
          "changed_part": "changed parts"}
SPANS = ["in the last {h} hours", "over the past {h} hours", "since {h} hours ago"]
SYNTH = {
    "yield": ["How many boards failed in the last {h} hours?", "What is the pass rate {when}?",
              "How many boards went to a person {when}?", "How many boards did we inspect {when}?",
              "How many boards passed {when}?", "Give me the yield for the last {h} hours",
              "How many boards needed checking by an operator {when}?", "Failed versus passed boards {when}?"],
    "defect_counts": ["Which defect is the most common {when}?", "How many {type} {when}?",
                      "What defects did we find in the last {h} hours?", "Are there more {type} or {type2} {when}?",
                      "What was the main problem {when}?", "List the defects found {when}"],
    "severity_counts": ["How many critical defects {when}?", "Split the defects {when} into critical, major and minor",
                        "How serious were the defects {when}?", "Were there more major or minor defects {when}?"],
    "board_history": ["Show the history of board {board}", "What happened to board {board}?",
                      "How many times was board {board} inspected?", "Look up board {board}"],
    "open_alerts": ["Are there any open alerts?", "Any drift alerts I should know about?",
                    "Is the station warning about anything?", "Show me the open alerts"],
}
WHEN = ["today", "this shift", "in the last {h} hours", "this week"]
ALERTS = ["short on 6 of the last 20 boards (3x the usual rate): check the etch line",
          "open repeating at the same spot on 5 boards: inspect the phototool",
          "boards sent to a person above the control limit for 2 groups of 20"]


def _questions(split: str) -> dict[str, list[str]]:
    y = yaml.safe_load((ROOT / "llm" / "ask_questions.yaml").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {q: [] for q in QIDS}
    for e in y[split]:
        out[e["query"]].append(e["q"])
    return out


def heldout_questions() -> list[str]:
    return [q for qs in _questions("heldout").values() for q in qs]


def _synth_question(qid: str, rng: random.Random) -> str:
    h = rng.choice([2, 3, 4, 6, 8, 12, 24])
    t1, t2 = rng.sample(TYPES, 2)
    return rng.choice(SYNTH[qid]).format(h=h, when=rng.choice(WHEN).format(h=h), type=PLURAL[t1], type2=PLURAL[t2],
                                         board=f"B{rng.randint(1, 99999):05d}")


def _rows(qid: str, question: str, rng: random.Random) -> list[dict]:
    if qid == "yield":
        boards = rng.choice([0, rng.randint(1, 12), rng.randint(10, 80)])
        a = rng.randint(0, boards)
        b = rng.randint(0, boards - a)
        return [{"boards": boards, "passed": a, "failed": b, "review": boards - a - b}]
    if qid == "defect_counts":
        types = list(dict.fromkeys(named_types(question) + rng.sample(TYPES, rng.randint(1, 5))))
        types = [t for t in types if rng.random() < 0.85] or types[:1]
        counts = sorted((rng.randint(1, 40) for _ in types), reverse=True)
        if len(counts) > 1 and rng.random() < 0.15:
            counts[1] = counts[0]                                    # a tie for the top
        return [{"defect": t, "n": n} for t, n in zip(types, counts)]
    if qid == "severity_counts":
        return [{"severity": s, "n": rng.randint(1, 50)} for s in ("critical", "major", "minor") if rng.random() < 0.8]
    if qid == "board_history":
        return [{"inspection": rng.randint(1, 900), "time": f"2026-11-{rng.randint(10, 20)} {rng.randint(8, 18)}:{rng.randint(10, 59)}:00",
                 "verdict": rng.choice(["PASS", "FAIL", "REVIEW"]), "defects": rng.randint(0, 9)}
                for _ in range(rng.choice([0, 1, 1, 2, 3]))]
    return [{"time": f"2026-11-{rng.randint(10, 20)} {rng.randint(8, 18)}:15:00", "message": m}
            for m in rng.sample(ALERTS, rng.randint(0, 2))]


SINGULAR = {"open": "open", "short": "short", "mouse_bite": "mouse bite", "spur": "spur",
            "spurious_copper": "copper island", "pin_hole": "pinhole", "missing_hole": "missing hole",
            "missing_pad": "missing pad", "hole_breakout": "hole breakout", "scratch": "scratch",
            "foreign_object": "foreign object", "missing_part": "missing part", "extra_part": "extra part",
            "changed_part": "changed part"}


def _target(qid: str, f: dict, need: set[str], rng: random.Random) -> str:
    """A correct fill-in sentence. {field|noun} lets the station write "1 board" / "2 boards". Several phrasings per
    answer so the model learns to answer the question rather than one fixed sentence per question type."""
    pick = rng.choice
    span = pick(["in the last {hours|hour}", "over the last {hours|hour}", "in the past {hours|hour}"])
    if qid == "yield":
        if "pass_rate_pct" in need:
            s = pick(["The pass rate is {pass_rate_pct}%: {passed} of {boards|board} passed.",
                      "{pass_rate_pct}% of boards passed " + span + " ({passed} of {boards}).",
                      "Yield " + span + " is {pass_rate_pct}%, with {passed} of {boards|board} passing.",
                      "{passed} of {boards|board} passed, a pass rate of {pass_rate_pct}%."])
            extra = sorted(need - {"pass_rate_pct", "passed", "boards"})
            if extra:
                s += " " + " and ".join({"failed": "{failed} failed", "review": "{review} went to a person"}[k]
                                        for k in extra).capitalize() + "."
            return s
        if need in ({"review"}, {"review", "boards"}):
            return pick(["{review} of the {boards|board} went to a person for checking.",
                         "A person was asked to check {review} of the {boards|board}.",
                         "Out of {boards|board} inspected, {review} went to a person.",
                         "{review} went to an operator for a second look, out of {boards|board}."])
        if need in ({"failed"}, {"failed", "boards"}):
            return pick(["{failed} of the {boards|board} inspected failed.", "Out of {boards|board}, {failed} failed.",
                         "{failed} failed " + span + ", out of {boards|board}."])
        if need in ({"passed"}, {"passed", "boards"}):
            return pick(["{passed} of the {boards|board} inspected passed.", "Out of {boards|board}, {passed} passed.",
                         "{passed} passed " + span + ", out of {boards|board}."])
        if need == {"boards"}:
            return pick(["The station inspected {boards|board} " + span + ".",
                         "The station inspected {boards|board}: {passed} passed, {failed} failed and {review} went to a person.",
                         "{boards|board} went through the station " + span + "."])
        return pick(["Of {boards|board}, {passed} passed, {failed} failed and {review} went to a person.",
                     "{passed} passed and {failed} failed out of {boards|board}; {review} went to a person."])
    if qid == "defect_counts":
        named = [k[len("count_"):] for k in sorted(need) if k.startswith("count_")]
        if "comparison" in need:
            a, b = named[:2]
            return pick([f"There were {{comparison}} " + span + f": {{count_{a}|{SINGULAR[a]}}} and {{count_{b}|{SINGULAR[b]}}}.",
                         f"{{comparison}}: {{count_{a}|{SINGULAR[a]}}} against {{count_{b}|{SINGULAR[b]}}}.",
                         f"The station found {{comparison}}, {{count_{a}|{SINGULAR[a]}}} and {{count_{b}|{SINGULAR[b]}}}."])
        if named:
            body = " and ".join(f"{{count_{t}|{SINGULAR[t]}}}" for t in named)
            return pick(["Found " + span + ": " + body + ".", "The station found " + body + " " + span + ".",
                         "There were " + body + " " + span + "."])
        if "second_count" in f and f["second_count"] == f["top_count"]:
            return pick(["{top_defect} and {second_defect} are the most common, {top_count} each.",
                         "The most common are {top_defect} and {second_defect}, with {top_count} each."])
        if "top_defect" in need:
            s = pick(["The most common defect is {top_defect}, found {top_count|time}.",
                      "{top_defect} is the most frequent defect, with {top_count} found.",
                      "Most of the defects were {top_defect}: {top_count} of {total_defects}.",
                      "The main problem " + span + " was {top_defect}, with {top_count} found."])
            if "second_defect" in f and rng.random() < 0.5:
                s += " Next is {second_defect} with {second_count}."
            return s
        return pick(["{total_defects|defect} of {n_types|type} were found " + span + "; the most common is {top_defect} ({top_count}).",
                     "The station found {total_defects|defect} " + span + ", mostly {top_defect} ({top_count})."])
    if qid == "severity_counts":
        if "comparison" in need:
            a, b = sorted(need - {"comparison"})[:2]
            return pick([f"There were {{comparison}} defects " + span + f": {{{a}}} {a} and {{{b}}} {b}.",
                         f"{{comparison}} defects: {{{a}}} {a} against {{{b}}} {b}.",
                         f"The station found {{comparison}} defects, {{{a}}} {a} and {{{b}}} {b}."])
        if need >= {"critical", "major", "minor"} or not need:
            return pick(["{critical} critical, {major} major and {minor} minor defects were found " + span + ".",
                         "Of {total_defects|defect} " + span + ", {critical} were critical, {major} major and {minor} minor.",
                         "By severity " + span + ": {critical} critical, {major} major, {minor} minor."])
        body = " and ".join(f"{{{s}}} {s}" for s in ("critical", "major", "minor") if s in need)
        return pick([span[0].upper() + span[1:] + ": " + body + ", out of {total_defects|defect}.",
                     "There were " + body + " defects " + span + ", out of {total_defects}."])
    if qid == "board_history":
        return pick(["Board {board} was inspected {times_inspected|time}; the last verdict was {last_verdict} "
                     "with {last_defects|defect}.",
                     "{board} has been through the station {times_inspected|time}, ending with {last_verdict} "
                     "and {last_defects|defect}.",
                     "{board} was inspected {times_inspected|time}. Its last result was {last_verdict}, "
                     "with {last_defects|defect}."])
    return pick(["Open alerts: {n_alerts}. The latest says: {latest_alert}",
                 "There are {n_alerts|open alert}; the latest says: {latest_alert}",
                 "The station has {n_alerts|open alert}. The newest: {latest_alert}"])


def examples(n: int, seed: int = 0) -> list[dict]:
    rng, tuning, held = random.Random(seed), _questions("tuning"), set(heldout_questions())
    out = []
    while len(out) < n:
        qid = rng.choice(QIDS)
        q = rng.choice(tuning[qid]) if tuning[qid] and rng.random() < 0.4 else _synth_question(qid, rng)
        if q in held:
            continue
        params = {**QUERIES[qid]["params"], **question_params(q, qid)}
        rows = _rows(qid, q, rng)
        if not rows:
            continue                              # the station answers an empty result without Llama
        facts = facts_for(qid, rows, params, q)
        if nothing_to_say(qid, facts):
            continue                              # the station answers these itself, without Llama
        out.append({"qid": qid, "question": q, "facts": facts,
                    "target": _target(qid, facts, required_fields(q, qid), rng)})
    return out


def training_rows(ex: dict) -> dict:
    return {"prompt": template_prompt(ex["question"], ex["facts"]), "completion": ex["target"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from llm.data import gen_why
    for name, exs, whys in (("sentences_train", examples(a.n, 21), gen_why.examples(a.n // 3, 31)),
                            ("sentences_dev", examples(a.n // 10, 22), gen_why.examples(a.n // 30, 32))):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps({**e, **training_rows(e)}) + "\n")
            for e in whys:            # the "why" line under each cause (pcbinspect/why.py)
                f.write(json.dumps({**e, "qid": "why", **gen_why.training_rows(e)}) + "\n")
        print(name, len(exs), "+", len(whys), "why")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

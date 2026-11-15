"""Audit of the Llama training data: what the generators actually produce, before any training run.

Cause data: examples per defect type and per true cause, clue combinations and their reuse, how informative the
labels are (the best possible top-1 given the clues), causes the clues can never single out, how often each clue
is unknown, and whether held-out combinations leak. Sentence data: examples per question type, repeated targets,
held-out question leakage. Writes a JSON summary and a readable sample for reading by hand.

    python -m llm.data.audit --out results/measured/data_audit      [--n 6000 --n-sent 3000 --sample 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path

from llm.data.gen_cause import alt_examples, examples, held_out
from llm.data.gen_sentences import examples as sent_examples
from llm.data.gen_sentences import heldout_questions, training_rows as sent_rows
from pcbinspect.cause_prompt import clue_text
from pcbinspect.causes import CauseModel
from pcbinspect.kb import default_kb


def _q(xs, ps=(0.1, 0.5, 0.9)):
    s = sorted(xs)
    return {f"p{int(p * 100)}": round(s[min(len(s) - 1, int(p * len(s)))], 3) for p in ps}


def audit_cause(n: int, kb, cm: CauseModel) -> dict:
    train, dev = examples(n, 11, "train", kb), examples(n // 10, 12, "dev", kb)
    test, alt = examples(n // 10, 13, "test", kb), alt_examples(n // 10, 14, kb)
    tgt = lambda e: next(d for d in e["record"]["defects"] if d["id"] == e["defect_id"])
    by_type = collections.Counter(tgt(e)["type"] for e in train)
    by_cause = collections.Counter(e["true_cause"] for e in train)
    all_causes = {c for t in kb.defects for c in cm.causes(t)}
    combos = collections.Counter(e["combo"] for e in train)
    test_combos = {e["combo"] for e in test}
    top = lambda e: max(e["posterior"], key=e["posterior"].get)

    # the label's ceiling: how often the most probable cause given the clues is the cause that made them
    per_type = collections.defaultdict(lambda: [0, 0])
    for e in train:
        t = per_type[tgt(e)["type"]]
        t[0] += top(e) == e["true_cause"]
        t[1] += 1
    # causes that are never the most probable for ANY clue combination seen: the model can never name them first
    winners = {top(e) for e in train + test}
    never_top = sorted(all_causes - winners)
    pmax = [max(e["posterior"].values()) for e in train]
    ent = [-sum(p * math.log(p) for p in e["posterior"].values() if p > 0) / math.log(max(2, len(e["posterior"])))
           for e in train]
    unknown = collections.Counter()
    for e in train:
        ev = tgt(e)["evidence"]
        for k in ("repeat_same_spot", "near_edge", "trend"):
            unknown[k] += ev.get(k) == "unknown"
    prov = collections.Counter(cm.evidence[e["true_cause"]]["provenance"] for e in train)
    return {
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test), "alt": len(alt)},
        "examples_per_defect_type": dict(by_type.most_common()),
        "examples_per_true_cause": {"min": min(by_cause.values()), "max": max(by_cause.values()),
                                    "causes_with_none": sorted(all_causes - set(by_cause)),
                                    "causes_under_20": sorted(c for c, k in by_cause.items() if k < 20)},
        "clue_combinations": {"train_distinct": len(combos), "test_distinct": len(test_combos),
                              "examples_per_combination": _q(list(combos.values())),
                              "test_combinations_in_train": len(test_combos & set(combos)),
                              "dev_combinations_in_train": len({e["combo"] for e in dev} & set(combos)),
                              "train_held_out_flag": sum(held_out(c) for c in combos)},
        "labels": {"bayes_top1_on_train": round(sum(top(e) == e["true_cause"] for e in train) / len(train), 3),
                   "bayes_top1_by_defect_type": {t: round(a / b, 3) for t, (a, b) in sorted(per_type.items())},
                   "max_probability": _q(pmax), "normalised_entropy": _q(ent),
                   "share_with_max_p_under_0.5": round(sum(p < 0.5 for p in pmax) / len(pmax), 3),
                   "causes_never_most_probable": never_top,
                   "causes_never_most_probable_share": f"{len(never_top)}/{len(all_causes)}"},
        "unknown_clue_rate": {k: round(v / len(train), 3) for k, v in unknown.items()},
        "true_cause_provenance": dict(prov),
    }, train


def audit_sentences(n: int) -> tuple[dict, list]:
    exs = sent_examples(n, 0)
    held = set(heldout_questions())
    by_q = collections.Counter(e["qid"] for e in exs)
    tg = collections.Counter(e["target"] for e in exs)
    qs = collections.Counter(e["question"] for e in exs)
    lens = [len(sent_rows(e)["prompt"]) for e in exs]
    return {"size": len(exs), "examples_per_question_type": dict(by_q.most_common()),
            "distinct_questions": len(qs), "distinct_targets": len(tg),
            "most_repeated_targets": tg.most_common(5),
            "held_out_questions_in_training": sum(q in held for q in qs),
            "prompt_chars": _q(lens)}, exs


def readable(train, sents, kb, k: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = ["# Training data sample (for reading by hand)\n",
           "Each cause example: the clues the model sees for the defect, the cause the generator used, and the label "
           "(probability over causes). Mark anything a process engineer would disagree with.\n"]
    for i, e in enumerate(rng.sample(train, k), 1):
        d = next(d for d in e["record"]["defects"] if d["id"] == e["defect_id"])
        post = sorted(e["posterior"].items(), key=lambda x: -x[1])[:3]
        out.append(f"## {i}. {d['type']}\n- {clue_text(d['evidence'])}\n"
                   f"- generator's cause: {kb.by_id[e['true_cause']]['short']} ({e['true_cause']})\n"
                   "- label: " + ", ".join(f"{kb.by_id[c]['short']} {p:.2f}" for c, p in post) + "\n")
    out.append("\n# Sentence sample\n")
    for e in rng.sample(sents, min(20, len(sents))):
        out.append(f"- **{e['question']}** ({e['qid']})  \n  facts: `{json.dumps(e['facts'])[:220]}`  \n"
                   f"  target: {e['target']}\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--n-sent", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=50)
    a = ap.parse_args()
    kb = default_kb()
    cm = CauseModel(kb)
    cause, train = audit_cause(a.n, kb, cm)
    sent, sents = audit_sentences(a.n_sent)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps({"cause": cause, "sentences": sent}, indent=1), encoding="utf-8")
    (out / "sample.md").write_text(readable(train, sents, kb, a.sample), encoding="utf-8")
    print(json.dumps({"cause": cause, "sentences": sent}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How well does a scorer choose the cause of a defect from the station's clues?

Scorers:
  prior      the cause model's prior only (no clues): the floor
  bayes      the cause model (kb/evidence.yaml) with the clues: the plain lookup baseline, and on generator 1 the
             exact answer (the Bayes ceiling)
  llama      a Llama .pte through executorch.runtime (pcbinspect/llm_logits.py): one yes/no question per cause,
             probabilities normalised over the defect's causes (what the station does)

Test sets: generator 1's held-out clue combinations ("test"), generator 2 ("alt", built differently), and the
published real cases (llm/data/real_cases.yaml).

    python llm/eval_cause.py --scorer bayes --n 600 --out results/measured/cause_bayes.json
    python llm/eval_cause.py --scorer llama --pte <model.pte> --tok <tokenizer.model> --n 300 --out ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.data.gen_cause import alt_examples, examples  # noqa: E402
from pcbinspect.cause_prompt import build_evidence_prompt, cause_question  # noqa: E402
from pcbinspect.causes import CauseModel, features_from  # noqa: E402
from pcbinspect.kb import default_kb  # noqa: E402


def real_examples(kb) -> list[dict]:
    cm = CauseModel(kb)
    out = []
    for c in yaml.safe_load((ROOT / "llm" / "data" / "real_cases.yaml").read_text(encoding="utf-8"))["cases"]:
        d = {"id": "D1", **c["defect"], "confidence": 0.9, "evidence": {**c["evidence"], "co_occurring": []}}
        out.append({"id": c["id"], "generator": "real", "record": {"board_id": c["id"], "verdict": "FAIL",
                    "defects": [d]}, "defect_id": "D1", "true_cause": c["true_cause"], "variant": 0,
                    "posterior": cm.posterior(d["type"], features_from(d["evidence"]))})
    return out


def probs_prior(ex, cm, kb):
    cls = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])["type"]
    return {c: cm.evidence[c]["prior"] for c in cm.causes(cls)}


def probs_bayes(ex, cm, kb):
    d = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])
    return cm.posterior(d["type"], features_from(d["evidence"]))


def make_llama(pte: str, tok: str, temperature: float = 1.0):
    from pcbinspect.llm_logits import LogitLlama
    be = LogitLlama(pte, tok)

    def probs(ex, cm, kb):
        d = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])
        be.reset()
        be.feed(build_evidence_prompt(ex["record"], kb, variant=0))
        lo = {c: be.yes_no(cause_question(d["id"], c, kb)) for c in cm.causes(d["type"])}
        m = max(lo.values())
        e = {c: math.exp((v - m) / temperature) for c, v in lo.items()}
        z = sum(e.values())
        return {c: v / z for c, v in e.items()}
    return probs, be.model


def score(exs, probs_fn, cm, kb, abstain_gap: float = 0.15) -> dict:
    n = top1 = top2 = agree = handle_ok = handle_n = within_ok = within_n = abst = abst_right = 0
    brier, ece_bins = 0.0, [[0, 0.0, 0] for _ in range(10)]
    for ex in exs:
        p = probs_fn(ex, cm, kb)
        order = sorted(p, key=p.get, reverse=True)
        t = ex["true_cause"]
        n += 1
        top1 += order[0] == t
        top2 += t in order[:2]
        bayes_best = max(ex["posterior"], key=ex["posterior"].get)
        agree += order[0] == bayes_best
        brier += sum((p[c] - (c == t)) ** 2 for c in p)
        b = min(9, int(p[order[0]] * 10))
        ece_bins[b][0] += 1
        ece_bins[b][1] += p[order[0]]
        ece_bins[b][2] += order[0] == t
        step = lambda c: kb.by_id[c].get("step")
        if any(step(c) == "handling" for c in p) and any(step(c) != "handling" for c in p):
            handle_n += 1
            handle_ok += (step(order[0]) == "handling") == (step(t) == "handling")
        proc = [c for c in p if step(c) != "handling"]
        if step(t) != "handling" and len(proc) >= 2:
            within_n += 1
            within_ok += order[0] == t
        if len(order) > 1 and p[order[0]] - p[order[1]] < abstain_gap:
            abst += 1
            abst_right += t in order[:2]
    ece = sum(abs(s / max(c, 1) - k / max(c, 1)) * c for c, s, k in ece_bins) / max(n, 1)
    answered = n - abst
    return {"n": n, "top1": top1 / n, "top2": top2 / n, "agree_with_bayes": agree / n, "brier": brier / n,
            "ece": ece, "handling_vs_process": handle_ok / max(handle_n, 1), "handling_vs_process_n": handle_n,
            "within_process": within_ok / max(within_n, 1), "within_process_n": within_n,
            "abstain_rate": abst / n, "abstained_true_in_top2": abst_right / max(abst, 1),
            "answered": answered}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=("prior", "bayes", "llama"), required=True)
    ap.add_argument("--pte")
    ap.add_argument("--tok")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kb = default_kb()
    cm = CauseModel(kb)
    model = a.scorer
    if a.scorer == "llama":
        fn, model = make_llama(a.pte, a.tok)
    else:
        fn = probs_prior if a.scorer == "prior" else probs_bayes
    sets = {"test": examples(a.n, 13, "test", kb), "alt": alt_examples(a.n, 14, kb), "real": real_examples(kb)}
    out = {"scorer": a.scorer, "model": model, "n_per_set": a.n, "sets": {}}
    for name, exs in sets.items():
        t0 = time.perf_counter()
        out["sets"][name] = {**score(exs, fn, cm, kb), "seconds": round(time.perf_counter() - t0, 1)}
        print(name, json.dumps(out["sets"][name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
